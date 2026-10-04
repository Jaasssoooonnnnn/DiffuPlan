import logging
from contextlib import nullcontext
import math
import os
import re
import warnings
import gc
import time
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer

try:
    from dinfer_sync_payload import trace_rl_prepare_rank_local_sync_payload
except ImportError:
    from .dinfer_sync_payload import trace_rl_prepare_rank_local_sync_payload

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", module="torch._inductor")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def _get_patch_block_diffusion_llm_generate():
    try:
        from dinfer_patch import patch_block_diffusion_llm_generate
    except ImportError:
        from .dinfer_patch import patch_block_diffusion_llm_generate
    return patch_block_diffusion_llm_generate


def _get_compile_patch_fns():
    try:
        from dinfer_compile_patch import (
            apply_compile_exclusion_patches,
            patch_torch_compile_config,
        )
    except ImportError:
        from .dinfer_compile_patch import (
            apply_compile_exclusion_patches,
            patch_torch_compile_config,
        )
    return apply_compile_exclusion_patches, patch_torch_compile_config


def _apply_dinfer_compile_patches() -> None:
    # Keep this lazy: importing dinfer_compile_patch may import dInfer/sglang
    # internals, so only do it after dInfer runtime modules are already needed.
    os.environ.setdefault("DINFER_DISABLE_COORDINATE_DESCENT_TUNING", "1")
    apply_compile_exclusion_patches, patch_torch_compile_config = _get_compile_patch_fns()
    patch_torch_compile_config()
    if os.environ.get("DINFER_DISABLE_COMPILED_SPARSE_MOE", "0") == "1":
        apply_compile_exclusion_patches()

_BUCKET_SIZE = 32
_RETRIEVAL_LIMIT_FAILURE = "Failure: Max retrieval count exceeded."
_TAG_PARSE_CARRY_CHARS = 4096
_ANSWER_FORMAT_ERROR_SENTINEL = "__trace_rl_answer_format_error__"
_MALFORMED_ANSWER_CLOSE_SENTINEL = _ANSWER_FORMAT_ERROR_SENTINEL
_RE_SEARCH_BLOCK = re.compile(r"<search\b[^>]*>(.*?)</search\s*>", re.DOTALL | re.IGNORECASE)
_RE_ANSWER_BLOCK = re.compile(r"<answer\b[^>]*>(.*?)</answer\s*>", re.DOTALL | re.IGNORECASE)
_RE_EMPTY_SEARCH_BLOCK = re.compile(r"<search\b[^>]*>\s*</search\s*>", re.DOTALL | re.IGNORECASE)
_RE_XML_BLOCK = re.compile(r"<([A-Za-z][\w:.-]*)\b[^>]*>.*?</\1\s*>", re.DOTALL)
_ALLOWED_GENERATED_XML_TAGS = {"think", "search", "answer"}
_RE_NON_ANSWER_CONTROL_TAG = re.compile(
    r"</?(?:search|information|think|plan|replan|step)\b[^>]*>",
    re.IGNORECASE,
)

logging.getLogger("dinfer_patch").setLevel(logging.WARNING)

_MODEL_RUNTIME_DEPS: Optional[Dict[str, Any]] = None
_TRACE_RL_DECODER_CLASS: Optional[type] = None
_SERVER_ARGS_CLS: Optional[type] = None
_INITIALIZE_MOE_CONFIG: Optional[Any] = None


def _patch_llada2_sglang_apply_state_dicts_inplace() -> None:
    from dinfer.model.modeling_llada2_moe_sglang import (
        FusedMoE,
        LLaDA2SGLangLM,
        LLaDA2SparseMoeBlock,
        PPMissingLayer,
    )
    from sglang.srt.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )

    if getattr(LLaDA2SGLangLM, "_trace_rl_apply_state_dicts_patch", False):
        return

    def _copy_or_rebind_tensor(
        owner: Any,
        name: str,
        live_tensor: torch.Tensor,
        loaded_weight: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        target = loaded_weight.to(device=live_tensor.device, dtype=live_tensor.dtype)
        try:
            live_tensor.copy_(target)
            return None
        except Exception as exc:
            owner._last_apply_used_rebind = True
            owner._last_apply_inplace_guard_failures.append(
                f"{name}: {type(exc).__name__}: {exc}"
            )
            return target

    def _apply_state_dicts_trace_rl(
        self,
        weights: Iterable[Tuple[str, torch.Tensor]],
        is_nextn: bool = False,
    ):
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )

        params_dict = dict(self.named_parameters())

        new_state_dict_keys = weights.keys()
        self_state_dict_keys = params_dict.keys()
        unused_keys = []
        for key in new_state_dict_keys:
            if key not in self_state_dict_keys:
                unused_keys.append(key)

        not_inited_keys = []
        for key in self_state_dict_keys:
            if key not in new_state_dict_keys:
                not_inited_keys.append(key)

        print("unused_keys", unused_keys)
        print("not_inited_keys", not_inited_keys)

        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        for key, value in weights.items():
            if key not in params_dict:
                print(f"not match:{key}")
                continue
            if self.quant_config is not None:
                if value.dim() == 0 and params_dict[key].dim() == 1:
                    if params_dict[key].shape[0] == 1:
                        value = value.unsqueeze(0)
                    elif params_dict[key].shape[0] == 2:
                        value = value.unsqueeze(0).repeat(2)
                    elif params_dict[key].shape[0] == 3:
                        value = value.unsqueeze(0).repeat(3)
                    weights[key] = value
            if value.shape != params_dict[key].shape:
                if not re.search(r"query_key_value.weight$", key):
                    mismatch_dim = 0 if value.shape[0] != params_dict[key].shape[0] else 1
                    if mismatch_dim == 0:
                        part_size = params_dict[key].shape[0]
                        weights[key] = (
                            value[tp_rank * part_size : (tp_rank + 1) * part_size]
                            if self.quant_config is None
                            else value[
                                tp_rank * part_size : (tp_rank + 1) * part_size
                            ].contiguous()
                        )
                    else:
                        part_size = params_dict[key].shape[1]
                        weights[key] = (
                            value[:, tp_rank * part_size : (tp_rank + 1) * part_size]
                            if self.quant_config is None
                            else value[
                                :, tp_rank * part_size : (tp_rank + 1) * part_size
                            ].contiguous()
                        )
                    if weights[key].shape != params_dict[key].shape:
                        print(
                            "shape mismatch fixed:",
                            key,
                            weights[key].shape,
                            params_dict[key].shape,
                        )
                else:
                    hidden_size = self.config.hidden_size
                    total_num_heads = self.config.num_attention_heads
                    total_kv_heads = self.config.num_key_value_heads
                    q_dim = hidden_size
                    q_part = q_dim // tp_size
                    q_weight = value[tp_rank * q_part : (tp_rank + 1) * q_part]
                    if tp_size > total_kv_heads:
                        n_replica = tp_size // total_kv_heads
                        kv_dim = hidden_size * total_kv_heads // total_num_heads
                        kv_part = kv_dim // total_kv_heads
                        k_weight = value[
                            q_dim + (tp_rank // n_replica) * kv_part : q_dim + ((tp_rank // n_replica) + 1) * kv_part
                        ]
                        v_weight = value[
                            q_dim + kv_dim + (tp_rank // n_replica) * kv_part : q_dim
                            + kv_dim
                            + ((tp_rank // n_replica) + 1) * kv_part
                        ]
                    else:
                        kv_dim = hidden_size * total_kv_heads // total_num_heads
                        kv_part = kv_dim // tp_size
                        k_weight = value[q_dim + tp_rank * kv_part : q_dim + (tp_rank + 1) * kv_part]
                        v_weight = value[
                            q_dim + kv_dim + tp_rank * kv_part : q_dim + kv_dim + (tp_rank + 1) * kv_part
                        ]
                    weights[key] = torch.cat([q_weight, k_weight, v_weight], dim=0)
                    assert weights[key].shape == params_dict[key].shape

        params_dict = dict(self.named_parameters())
        buffer_dict = dict(self.named_buffers())
        sync_inplace_mode = bool(getattr(self, "_trace_rl_sync_inplace_mode", False))
        self._last_apply_used_rebind = False
        self._last_apply_inplace_guard_failures = []

        for name, loaded_weight in weights.items():
            if name in params_dict:
                param = params_dict[name]
                if sync_inplace_mode:
                    rebound = _copy_or_rebind_tensor(self, name, param.data, loaded_weight)
                    if rebound is not None:
                        param.data = rebound
                else:
                    param.data = loaded_weight
            elif name in buffer_dict:
                buffer = buffer_dict[name]
                if sync_inplace_mode:
                    rebound = _copy_or_rebind_tensor(self, name, buffer.data, loaded_weight)
                    if rebound is not None:
                        buffer.data = rebound
                else:
                    buffer.data = loaded_weight
            else:
                print("params not matching:", name)

        if not is_nextn:
            self.routed_experts_weights_of_layer = {
                layer_id: layer.mlp.get_moe_weights()
                for layer_id, layer in enumerate(self.model.layers)
                if not isinstance(layer, PPMissingLayer)
                and isinstance(layer.mlp, LLaDA2SparseMoeBlock)
            }

    LLaDA2SGLangLM.apply_state_dicts = _apply_state_dicts_trace_rl
    LLaDA2SGLangLM._trace_rl_apply_state_dicts_patch = True


def _make_trace_rl_threshold_parallel_decoder_class(
    threshold_parallel_decoder_cls: Any,
    parallel_strategy_mod: Any,
):
    class TraceRLThresholdParallelDecoder(threshold_parallel_decoder_cls):
        """Repo-local decoder shim for deterministic zero-temperature compiled inference."""

        def __init__(
            self,
            temperature,
            threshold,
            remasking="low_confidence",
            mask_id=126336,
            eos_id=126081,
            use_float64=False,
            deterministic_zero_temp=False,
        ):
            super().__init__(
                temperature=temperature,
                threshold=threshold,
                remasking=remasking,
                mask_id=mask_id,
                eos_id=eos_id,
                use_float64=use_float64,
            )
            self.deterministic_zero_temp = bool(deterministic_zero_temp)
            self._trace_rl_logged_deterministic_zero_temp = False
            self.banned_token_ids: set[int] = set()

        def set_banned_token_ids(self, token_ids: Optional[Iterable[int]]) -> None:
            self.banned_token_ids = {int(token_id) for token_id in (token_ids or [])}

        def _apply_banned_tokens(self, logits: torch.Tensor) -> torch.Tensor:
            if not self.banned_token_ids:
                return logits
            work = logits.clone()
            for token_id in self.banned_token_ids:
                if 0 <= int(token_id) < int(work.shape[-1]):
                    work[..., int(token_id)] = float("-inf")
            return work

        def _use_deterministic_zero_temp(self) -> bool:
            return self.deterministic_zero_temp and math.isclose(float(self.temperature), 0.0, abs_tol=0.0)

        def _maybe_log_deterministic_zero_temp(self, iter_threshold: float) -> None:
            if not self._use_deterministic_zero_temp() or self._trace_rl_logged_deterministic_zero_temp:
                return
            print(
                "[dInfer] enabling deterministic zero-temp threshold decoding "
                f"(threshold={float(iter_threshold):.4f})",
                flush=True,
            )
            self._trace_rl_logged_deterministic_zero_temp = True

        def _get_transfer_index(
            self,
            logits: torch.Tensor,
            mask_index: torch.Tensor,
            curr_x: torch.Tensor,
            iter_threshold: float,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            logits = self._apply_banned_tokens(logits)
            if not self._use_deterministic_zero_temp():
                return parallel_strategy_mod.get_transfer_index_threshold(
                    logits,
                    self.temperature,
                    mask_index,
                    curr_x,
                    self.mask_id,
                    threshold=iter_threshold,
                    use_float64=self.use_float64,
                )

            self._maybe_log_deterministic_zero_temp(iter_threshold)

            x0 = torch.argmax(logits, dim=-1)
            prob_dtype = torch.float64 if self.use_float64 else torch.float32
            probs = F.softmax(logits.to(prob_dtype), dim=-1)
            x0_p = torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)

            valid_mask = mask_index & (x0 != self.mask_id)
            x0 = torch.where(valid_mask, x0, curr_x)

            neg_inf = torch.full_like(x0_p, float("-inf"))
            confidence = torch.where(valid_mask, x0_p, neg_inf)
            transfer_index = torch.zeros_like(valid_mask, dtype=torch.bool)

            max_confidence, max_indices = torch.max(confidence, dim=1)
            valid_rows = torch.nonzero(max_confidence > float("-inf"), as_tuple=False).flatten()
            if valid_rows.numel() == 0:
                return x0, transfer_index

            threshold_value = float(iter_threshold)
            high_conf_rows = valid_rows[max_confidence[valid_rows] >= threshold_value]
            if high_conf_rows.numel() > 0:
                transfer_index[high_conf_rows] = confidence[high_conf_rows] >= threshold_value

            low_conf_rows = valid_rows[max_confidence[valid_rows] < threshold_value]
            if low_conf_rows.numel() > 0:
                transfer_index[low_conf_rows, max_indices[low_conf_rows]] = True

            return x0, transfer_index

        def decode(self, logits, block_start, block_end, x, iter_threshold=None):
            if iter_threshold is None:
                iter_threshold = self.threshold
            mask_index = x[:, block_start:block_end] == self.mask_id
            assert mask_index.shape[1] == logits.shape[1]

            curr_x = x[:, block_start:block_end]
            x0, transfer_index = self._get_transfer_index(
                logits=logits,
                mask_index=mask_index,
                curr_x=curr_x,
                iter_threshold=float(iter_threshold),
            )
            transfer_index = torch.logical_and(transfer_index, mask_index)
            assert transfer_index.dtype == torch.bool
            x[:, block_start:block_end] = torch.where(transfer_index, x0, curr_x)
            parallel_strategy_mod.broadcast_if_needed(x.data)

        def batch_decode(self, logits, block_start, x, block_length, iter_threshold=None):
            if iter_threshold is None:
                iter_threshold = self.threshold

            batch_size, total_tokens = x.data.shape
            device = x.data.device
            offset = torch.arange(block_length, device=device).unsqueeze(0) + block_start.unsqueeze(1)
            x_block = torch.gather(x.data, 1, offset.clamp(max=total_tokens - 1))
            mask_index = x_block == self.mask_id

            x0, transfer_index = self._get_transfer_index(
                logits=logits,
                mask_index=mask_index,
                curr_x=x_block,
                iter_threshold=float(iter_threshold),
            )
            transfer_index = torch.logical_and(transfer_index, mask_index)
            x_updated = torch.where(transfer_index, x0, x_block)

            x_flat = x.data.view(-1)
            flat_idx = offset + torch.arange(batch_size, device=device).unsqueeze(1) * total_tokens
            x_flat[flat_idx] = x_updated
            parallel_strategy_mod.broadcast_if_needed(x.data)

    return TraceRLThresholdParallelDecoder


def _get_server_args_cls() -> type:
    global _SERVER_ARGS_CLS
    if _SERVER_ARGS_CLS is None:
        from sglang.srt.server_args import ServerArgs

        _SERVER_ARGS_CLS = ServerArgs
    return _SERVER_ARGS_CLS


def _get_initialize_moe_config() -> Any:
    global _INITIALIZE_MOE_CONFIG
    if _INITIALIZE_MOE_CONFIG is None:
        from sglang.srt.layers.moe import initialize_moe_config

        _INITIALIZE_MOE_CONFIG = initialize_moe_config
    return _INITIALIZE_MOE_CONFIG


def _get_model_runtime_deps() -> Dict[str, Any]:
    global _MODEL_RUNTIME_DEPS, _TRACE_RL_DECODER_CLASS
    if _MODEL_RUNTIME_DEPS is not None:
        return _MODEL_RUNTIME_DEPS

    from dinfer import BlockDiffusionLLM, BlockIteratorFactory, KVCacheFactory, ThresholdParallelDecoder
    from dinfer.decoding.diffusion_runner import ModelRunner
    import dinfer.decoding.parallel_strategy as dinfer_parallel_strategy
    from dinfer.model.modeling_llada2_moe_sglang import LLaDA2SGLangLM

    _apply_dinfer_compile_patches()
    _patch_llada2_sglang_apply_state_dicts_inplace()

    # Required TraceRL API extension: rollout generation passes return_step_map,
    # stop_token_sequences, and per-sequence generation budgets into dInfer.
    patch_block_diffusion_llm_generate = _get_patch_block_diffusion_llm_generate()
    patch_block_diffusion_llm_generate()

    # Apply collective/broadcast fixes after the dInfer modules are loaded.
    # Initialization may deliberately avoid importing these modules to prevent
    # their global side effects from leaking into benchmark/learner-only paths.
    engine_cls = globals().get("DInferRolloutEngine")
    if engine_cls is not None:
        engine_cls._patch_tp_collectives()
        engine_cls._patch_decoder_broadcast()

    if _TRACE_RL_DECODER_CLASS is None:
        _TRACE_RL_DECODER_CLASS = _make_trace_rl_threshold_parallel_decoder_class(
            ThresholdParallelDecoder,
            dinfer_parallel_strategy,
        )

    _MODEL_RUNTIME_DEPS = {
        "BlockDiffusionLLM": BlockDiffusionLLM,
        "BlockIteratorFactory": BlockIteratorFactory,
        "KVCacheFactory": KVCacheFactory,
        "ModelRunner": ModelRunner,
        "dinfer_parallel_strategy": dinfer_parallel_strategy,
        "LLaDA2SGLangLM": LLaDA2SGLangLM,
        "TraceRLThresholdParallelDecoder": _TRACE_RL_DECODER_CLASS,
    }
    return _MODEL_RUNTIME_DEPS


SEARCH_USER_RULES = (
    "Answer the given question. "
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> "
    "and it will return the top searched results between <information> and </information>. "
    "You can search as many times as your want. "
    "If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, "
    "without detailed illustrations. For example, <answer> Beijing </answer>."
)


@dataclass
class DInferRuntimeConfig:
    use_compile: bool = True
    deterministic_zero_temp: bool = True
    prefilling_limit: int = 128
    dist_timeout_sec: int = 120

class SearchClient:
    def __init__(self, url: str = "http://127.0.0.1:8000/retrieve", topk: int = 3):
        self.url = url
        self.topk = topk
        self.session = requests.Session()
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        retries = Retry(
            total=10,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["POST"],
            raise_on_status=False,
        )
        self.session.mount("http://", HTTPAdapter(max_retries=retries))

    def batch_search(self, queries: List[str], topk: Optional[int] = None) -> List[str]:
        if not queries:
            return []
        topk = topk or self.topk
        payload = {"queries": queries, "topk": topk, "return_scores": True}
        try:
            response = self.session.post(self.url, json=payload, timeout=60)
            response.raise_for_status()
            results = response.json()["result"]
            return [self._passages_to_string(r) for r in results]
        except Exception as e:
            print(f"[SearchClient Error] {type(e).__name__}: {e}", flush=True)
            return [""] * len(queries)

    @staticmethod
    def _passages_to_string(retrieval_result: List[dict]) -> str:
        formatted = ""
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item["document"]["contents"]
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            formatted += f"Doc {idx + 1}(Title: {title}) {text}\n"
        return formatted


def _extract_search_query(text: str) -> Optional[str]:
    matches = list(_RE_SEARCH_BLOCK.finditer(text))
    if matches:
        return matches[-1].group(1).strip()
    return None


def _extract_answer(text: str) -> str:
    matches = list(_RE_ANSWER_BLOCK.finditer(text))
    if matches:
        last_match = matches[-1]
        content = last_match.group(1).strip()
        start = last_match.start()
        pre_context = text[max(0, start - 15) : start]
        if "For example, " in pre_context and content == "Beijing":
            return ""
        return content
    return ""


def _extract_new_control_tags(carry: str, new_text: str) -> Tuple[str, Optional[str], str]:
    """
    Incrementally parse newly completed <search> / <answer> tags.
    This handles tags split across rollout iterations without re-triggering old matches.
    """
    combined = carry + new_text
    carry_len = len(carry)

    search_q: Optional[str] = None
    saw_empty_search = False
    for m in _RE_SEARCH_BLOCK.finditer(combined):
        if m.end() <= carry_len:
            continue
        q = m.group(1).strip()
        if q:
            search_q = q
        else:
            saw_empty_search = True

    answer_cand = ""
    saw_empty_answer = False
    for m in _RE_ANSWER_BLOCK.finditer(combined):
        if m.end() <= carry_len:
            continue
        content = m.group(1).strip()
        if not content:
            saw_empty_answer = True
            continue
        start = m.start()
        pre_context = combined[max(0, start - 15) : start]
        if "For example, " in pre_context and content == "Beijing":
            continue
        answer_cand = content

    # Guard against malformed answer closing tags.
    # If we see malformed answer syntax while no valid answer was extracted,
    # terminate early as format error to avoid long generate loops.
    if not answer_cand:
        if saw_empty_answer:
            answer_cand = _MALFORMED_ANSWER_CLOSE_SENTINEL

        answer_depth = 0
        saw_new_close_like_tag = False
        last_unclosed_open: Optional[Tuple[int, int]] = None
        for m in re.finditer(r"<answer\b[^>]*>|</answer[^>]*>|answer>", combined, re.IGNORECASE):
            tag = (m.group(0) or "").strip().lower()
            is_open = tag.startswith("<answer") and not tag.startswith("</")
            if is_open:
                answer_depth += 1
                last_unclosed_open = (int(m.start()), int(m.end()))
                continue

            is_close_like = tag.startswith("</answer")
            if is_close_like and answer_depth > 0:
                answer_depth -= 1
                if m.end() > carry_len:
                    # Even matched close without extractable content is a format issue.
                    saw_new_close_like_tag = True
                continue

            if tag == "answer>" and answer_depth > 0 and m.end() > carry_len:
                saw_new_close_like_tag = True
                break
            if tag == "answer>":
                continue

            if m.end() > carry_len:
                saw_new_close_like_tag = True
                break

        if not answer_cand and saw_new_close_like_tag:
            answer_cand = _MALFORMED_ANSWER_CLOSE_SENTINEL
        elif not answer_cand and last_unclosed_open is not None:
            _, open_end = last_unclosed_open
            next_control = _RE_NON_ANSWER_CONTROL_TAG.search(combined, pos=open_end)
            open_seen_before_this_turn = open_end <= carry_len
            has_boundary = next_control is not None
            if open_seen_before_this_turn or has_boundary:
                answer_cand = _MALFORMED_ANSWER_CLOSE_SENTINEL

    if not search_q and not answer_cand:
        empty_search_crossed_boundary = any(
            m.end() > carry_len
            for m in _RE_EMPTY_SEARCH_BLOCK.finditer(combined, max(0, carry_len - 64))
        )
        if saw_empty_search or empty_search_crossed_boundary:
            answer_cand = _MALFORMED_ANSWER_CLOSE_SENTINEL
        else:
            search_depth = 0
            saw_new_close_like_tag = False
            for m in re.finditer(r"<search\b[^>]*>|</search[^>]*>|search>", combined, re.IGNORECASE):
                tag = (m.group(0) or "").strip().lower()
                is_open = tag.startswith("<search") and not tag.startswith("</")
                if is_open:
                    search_depth += 1
                    continue

                is_close_like = tag.startswith("</search")
                if is_close_like and search_depth > 0:
                    search_depth -= 1
                    # A valid non-empty search block would already have set search_q.
                    if m.end() > carry_len:
                        saw_new_close_like_tag = True
                    continue

                if tag == "search>" and search_depth > 0 and m.end() > carry_len:
                    saw_new_close_like_tag = True
                    break
                if tag == "search>":
                    continue

                if m.end() > carry_len:
                    saw_new_close_like_tag = True
                    break

            if saw_new_close_like_tag:
                answer_cand = _MALFORMED_ANSWER_CLOSE_SENTINEL

    if not search_q and not answer_cand:
        for m in _RE_XML_BLOCK.finditer(combined):
            if m.end() <= carry_len:
                continue
            tag_name = str(m.group(1) or "").strip().lower()
            if tag_name and tag_name not in _ALLOWED_GENERATED_XML_TAGS:
                answer_cand = _MALFORMED_ANSWER_CLOSE_SENTINEL
                break

    if len(combined) > _TAG_PARSE_CARRY_CHARS:
        combined = combined[-_TAG_PARSE_CARRY_CHARS:]
    return combined, search_q, answer_cand


def _cut_eos_1d(ids_1d: torch.Tensor, eos_id: int) -> torch.Tensor:
    idx = (ids_1d == eos_id).nonzero(as_tuple=True)[0]
    if idx.numel() > 0:
        return ids_1d[: idx[0].item()]
    return ids_1d


def _build_search_prompt(question: str) -> str:
    return (
        f"<role>SYSTEM</role>You are a helpful assistant.<|role_end|>"
        f"<role>HUMAN</role>Question: {question}\n\n{SEARCH_USER_RULES}<|role_end|>"
        f"<role>ASSISTANT</role>"
    )


def _sanitize_generated_text(text: str, carry: str = "") -> str:
    """Apply rollout text postprocessing before appending history.

    Trim after completed control tags only when they close an actual preceding
    opening tag. This avoids dropping content on stray closing fragments while
    keeping token accounting aligned with the text that enters the trace.
    """
    text = text.replace("<|role_end|>", "")
    combined = (carry or "") + text
    carry_len = len(carry or "")
    best_end: Optional[int] = None
    for tag in ("search", "answer"):
        close_tag = f"</{tag}>"
        for m in re.finditer(re.escape(close_tag), combined):
            if m.end() <= carry_len:
                continue
            prefix = combined[: m.start()]
            open_count = len(re.findall(fr"<{tag}>", prefix))
            close_count = len(re.findall(fr"</{tag}>", prefix))
            if open_count > close_count:
                end_in_new = m.end() - carry_len
                best_end = end_in_new if best_end is None else min(best_end, end_in_new)
                break
    if best_end is not None:
        return text[:best_end]
    return text


def _extend_tag_parse_carry(carry: str, text: str) -> str:
    combined = (carry or "") + (text or "")
    if len(combined) > _TAG_PARSE_CARRY_CHARS:
        combined = combined[-_TAG_PARSE_CARRY_CHARS:]
    return combined


def _aligned_step_slice(
    turn_step_map_row: torch.Tensor,
    start: int,
    raw_gen_count: int,
    final_text: str,
    tokenizer: Any,
) -> List[int]:
    """Align step-map length to the postprocessed generated text."""
    raw_slice = turn_step_map_row[start : start + raw_gen_count]
    kept = len(tokenizer.encode(final_text, add_special_tokens=False))
    kept = min(max(kept, 0), int(raw_slice.shape[0]))
    return raw_slice[:kept].cpu().tolist()


_ENV_OBS_RE = re.compile(
    r"<\|role_end\|><role>HUMAN</role><information>.*?</information>\n?"
    r"<\|role_end\|><role>ASSISTANT</role>",
    re.DOTALL,
)


def _force_mask_environment_obs_tokens(
    response_text: str,
    step_map: List[int],
    tokenizer: Any,
) -> List[int]:
    """Ensure environment observation wrappers cannot become trainable tokens."""
    if not response_text or not step_map or "<|role_end|><role>HUMAN</role><information>" not in response_text:
        return step_map

    spans = [m.span() for m in _ENV_OBS_RE.finditer(response_text)]
    if not spans:
        return step_map

    try:
        enc = tokenizer(
            response_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offsets = enc.get("offset_mapping", [])
    except Exception:
        return step_map

    masked = list(step_map)
    span_idx = 0
    limit = min(len(masked), len(offsets))
    for tok_idx, (tok_start, tok_end) in enumerate(offsets[:limit]):
        while span_idx < len(spans) and spans[span_idx][1] <= tok_start:
            span_idx += 1
        if span_idx >= len(spans):
            break
        span_start, span_end = spans[span_idx]
        if tok_end > span_start and tok_start < span_end:
            masked[tok_idx] = -1
    return masked


class DInferRolloutEngine:
    def __init__(
        self,
        model_path: str,
        retriever_url: str,
        topk_search: int,
        max_batch_size: int,
        length: int,
        block_length: int,
        threshold: float,
        max_turns: int,
        max_obs_length: int,
        runtime_cfg: DInferRuntimeConfig,
        parallel_mode: str = "tp",
        unload_after_rollout: bool = False,
        expert_map_path: str = ".",
        tokenizer: Optional[Any] = None,
        model_config: Optional[Any] = None,
    ):
        self.model_path = model_path
        self.retriever_url = retriever_url
        self.topk_search = topk_search
        self.max_batch_size = max(1, int(max_batch_size))
        self.length = int(length)
        self.block_length = int(block_length)
        self.threshold = float(threshold)
        self.max_turns = int(max_turns)
        self.max_obs_length = int(max_obs_length)
        self.runtime_cfg = runtime_cfg
        self.parallel_mode = str(parallel_mode).strip().lower()
        if self.parallel_mode not in {"tp", "dp"}:
            raise ValueError(f"Unsupported parallel mode: {parallel_mode}. Expected 'tp' or 'dp'.")
        self.unload_after_rollout = bool(unload_after_rollout)
        self.expert_map_path = expert_map_path
        self._provided_tokenizer = tokenizer
        self._provided_model_config = model_config

        self.rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        self.world_size = (
            dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        )
        self.local_rank = int(os.environ.get("LOCAL_RANK", str(self.rank)))

        self.device: Optional[torch.device] = None
        self.tokenizer = None
        self.model = None
        self.model_runner = None
        self.decoder = None
        self.dllm = None
        self.search_client = None

        self._distributed = None
        self._model_config = None
        self._server_args = None
        self.initialized = False
        self.weights_loaded = False
        self.full_runtime_unload = os.environ.get(
            "DINFER_FULL_RUNTIME_UNLOAD", "true"
        ).strip().lower() in {"1", "true", "t", "yes", "y", "on"}
        self._last_rollout_perf: Dict[str, Any] = {"gen_elapsed_s": 0.0, "gen_tokens": 0.0}
        self._log_generate_calls = os.environ.get(
            "TRACE_RL_LOG_GEN_CALLS", "0"
        ).lower() in {"1", "true", "yes", "y", "on"}
        self._debug_tag_parse = os.environ.get(
            "TRACE_RL_DEBUG_TAG_PARSE", "0"
        ).lower() in {"1", "true", "yes", "y", "on"}
        self._debug_tag_parse_max_iters = max(
            0, int(os.environ.get("TRACE_RL_DEBUG_TAG_PARSE_MAX_ITERS", "0") or "0")
        )
        self._debug_tag_parse_ranks = {
            int(v.strip())
            for v in os.environ.get("TRACE_RL_DEBUG_TAG_PARSE_RANKS", "0").split(",")
            if v.strip().lstrip("-").isdigit()
        }
        self._dp_warmup_enabled = os.environ.get(
            "TRACE_RL_DP_WARMUP", "1"
        ).lower() in {"1", "true", "yes", "y", "on"}
        self._warmup_prompt_len = max(1, int(os.environ.get("TRACE_RL_WARMUP_PROMPT_LEN", "64")))
        self._warmup_gen_len = max(1, int(os.environ.get("TRACE_RL_WARMUP_GEN_LEN", "32")))
        self._stop_token_sequences: Optional[List[List[int]]] = None
        self._rollout_limits_logged = False
        self._last_sync_used_rebind = False
        self._last_sync_guard_failures: List[str] = []
        self._last_sync_requires_runtime_rebuild = False

    def _log_tag_parse_event(
        self,
        *,
        path: str,
        loop_iter: int,
        batch_idx: int,
        gen_token_count: int,
        new_text: str,
        search_q: Optional[str],
        answer_cand: str,
        retrieval_used: int,
        remaining_token_pool: int,
        no_search_tokens: int,
        stop_reason: str,
    ) -> None:
        if not self._debug_tag_parse:
            return
        if self._debug_tag_parse_ranks and self.rank not in self._debug_tag_parse_ranks:
            return
        if self._debug_tag_parse_max_iters and loop_iter > self._debug_tag_parse_max_iters:
            return
        snippet = (new_text or "").replace("\n", "\\n")
        if len(snippet) > 220:
            snippet = snippet[:110] + " ... " + snippet[-90:]
        search_preview = None
        if search_q is not None:
            search_preview = search_q.replace("\n", "\\n")
            if len(search_preview) > 120:
                search_preview = search_preview[:117] + "..."
        answer_preview = None
        if answer_cand:
            answer_preview = answer_cand.replace("\n", "\\n")
            if len(answer_preview) > 120:
                answer_preview = answer_preview[:117] + "..."
        print(
            f"[dInfer][rank{self.rank}][{path}] tag_parse "
            f"iter={loop_iter}, sample={batch_idx}, gen_tokens={gen_token_count}, "
            f"retrieval_used={retrieval_used}, remaining={remaining_token_pool}, "
            f"no_search={no_search_tokens}, search_q={search_preview!r}, "
            f"answer={answer_preview!r}, stop_reason={stop_reason!r}, text={snippet!r}",
            flush=True,
        )

    @staticmethod
    def _tensor_nbytes(t: torch.Tensor) -> int:
        return int(t.numel()) * int(t.element_size())

    def _log_local_cuda_mem(self, tag: str) -> None:
        if self.device is None or not torch.cuda.is_available():
            return
        try:
            allocated_mb = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
            reserved_mb = torch.cuda.memory_reserved(self.device) / (1024 ** 2)
            print(
                f"[dInfer][rank{self.rank}] cuda_mem {tag}: "
                f"allocated={allocated_mb:.1f}MB reserved={reserved_mb:.1f}MB",
                flush=True,
            )
        except Exception:
            # Best-effort diagnostics; never block rollout/unload flows.
            pass

    def _clear_routed_expert_aliases(self, reason: str = "") -> None:
        """Drop cached `.data` alias tensors that can pin old CUDA storages after `.to('cpu')`."""
        if self.model is None or not hasattr(self.model, "routed_experts_weights_of_layer"):
            return

        released_bytes = 0
        cached = getattr(self.model, "routed_experts_weights_of_layer", None)
        if isinstance(cached, dict):
            for value in cached.values():
                if isinstance(value, list):
                    for t in value:
                        if isinstance(t, torch.Tensor) and t.is_cuda:
                            released_bytes += self._tensor_nbytes(t)
                elif isinstance(value, torch.Tensor) and value.is_cuda:
                    released_bytes += self._tensor_nbytes(value)

        try:
            setattr(self.model, "routed_experts_weights_of_layer", None)
        except Exception:
            try:
                delattr(self.model, "routed_experts_weights_of_layer")
            except Exception:
                pass

        released_mb = released_bytes / (1024 ** 2)
        suffix = f" ({reason})" if reason else ""
        print(
            f"[dInfer][rank{self.rank}] cleared routed_experts_weights_of_layer{suffix}, "
            f"alias_cuda_estimate={released_mb:.1f}MB",
            flush=True,
        )

    def _clear_model_side_runtime_refs(self, reason: str = "") -> None:
        """Drop extra model-side runtime references that are safe to recreate before next rollout."""
        if self.model is None:
            return

        self._clear_routed_expert_aliases(reason=reason)

        # `h2e` is rebuilt by `init_h2e_module()` during sync/reload; clear to reduce dangling refs.
        if hasattr(self.model, "h2e"):
            try:
                setattr(self.model, "h2e", None)
            except Exception:
                pass

    def get_last_rollout_perf(self) -> Dict[str, Any]:
        """Return generate-only perf stats from the most recent `generate_rollouts` call."""
        return dict(self._last_rollout_perf)

    def _run_generate_warmup(self, mode: str) -> None:
        if self.device is None:
            return
        print(f"[dInfer][rank{self.rank}] warmup start (mode={mode})", flush=True)
        t_warmup = time.time()
        try:
            warmup_bs = min(self.max_batch_size, 4)
            warmup_ids = torch.randint(
                0,
                100000,
                (warmup_bs, self._warmup_prompt_len),
                dtype=torch.long,
                device=self.device,
            )
            self.dllm.generate(
                warmup_ids,
                gen_length=self._warmup_gen_len,
                block_length=self.block_length,
            )
            torch.cuda.synchronize(self.device)
            print(
                f"[dInfer][rank{self.rank}] warmup done in {time.time() - t_warmup:.2f}s "
                f"(mode={mode}, bs={warmup_bs}, prompt_len={self._warmup_prompt_len}, "
                f"gen_len={self._warmup_gen_len})",
                flush=True,
            )
        except Exception as e:
            print(f"[dInfer][rank{self.rank}] warmup warning: {e}", flush=True)

    def _log_generate_call(
        self,
        *,
        phase: str,
        path: str,
        loop_iter: int,
        batch_curr: int,
        max_len: int,
        requested_gen_len: int,
        valid_gen_len: int,
        per_seq_token_limits: List[int],
        elapsed_s: Optional[float] = None,
    ) -> None:
        if not self._log_generate_calls:
            return
        if per_seq_token_limits:
            min_limit = int(min(per_seq_token_limits))
            max_limit = int(max(per_seq_token_limits))
        else:
            min_limit = 0
            max_limit = 0
        msg = (
            f"[dInfer][rank{self.rank}][{path}] generate_{phase} "
            f"iter={loop_iter}, batch={batch_curr}, prompt_max={max_len}, "
            f"requested_gen={requested_gen_len}, aligned_gen={valid_gen_len}, "
            f"per_seq_limit_min={min_limit}, per_seq_limit_max={max_limit}"
        )
        if elapsed_s is not None:
            msg += f", elapsed={elapsed_s:.2f}s"
        print(msg, flush=True)

    @staticmethod
    def _new_rollout_diag() -> Dict[str, Any]:
        return {
            "loop_iters": 0,
            "generate_calls": 0,
            "active_batch_total": 0,
            "active_batch_max": 0,
            "active_singleton_iters": 0,
            "active_multi_iters": 0,
            "prompt_max_total": 0,
            "prompt_max_max": 0,
            "requested_gen_total": 0,
            "requested_gen_max": 0,
            "aligned_gen_total": 0,
            "aligned_gen_max": 0,
            "search_calls": 0,
            "search_queries_total": 0,
            "search_batch_max": 0,
            "search_elapsed_s": 0.0,
            "search_tag_events": 0,
            "retrieval_limit_events": 0,
            "answer_format_error_events": 0,
            "answer_close_only_events": 0,
            "slowest_generate_calls": [],
            "slowest_search_calls": [],
            "stop_reason_counts": {},
            "max_generated_tokens_per_sample": 0,
            "max_retrieval_used_per_sample": 0,
        }

    @staticmethod
    def _record_top_slowest(
        entries: List[Dict[str, Any]],
        item: Dict[str, Any],
        *,
        key: str = "elapsed_s",
        keep_top_k: int = 3,
    ) -> None:
        entries.append(item)
        entries.sort(key=lambda v: float(v.get(key, 0.0)), reverse=True)
        del entries[keep_top_k:]

    def _merge_rollout_diag(
        self, base_diag: Optional[Dict[str, Any]], new_diag: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        merged = dict(base_diag or self._new_rollout_diag())
        if not isinstance(new_diag, dict) or not new_diag:
            return merged

        int_keys = (
            "loop_iters",
            "generate_calls",
            "active_batch_total",
            "active_singleton_iters",
            "active_multi_iters",
            "prompt_max_total",
            "requested_gen_total",
            "aligned_gen_total",
            "search_calls",
            "search_queries_total",
            "search_tag_events",
            "retrieval_limit_events",
            "answer_format_error_events",
            "answer_close_only_events",
        )
        max_keys = (
            "active_batch_max",
            "prompt_max_max",
            "requested_gen_max",
            "aligned_gen_max",
            "search_batch_max",
            "max_generated_tokens_per_sample",
            "max_retrieval_used_per_sample",
        )
        float_keys = ("search_elapsed_s",)

        for key in int_keys:
            merged[key] = int(merged.get(key, 0)) + int(new_diag.get(key, 0))
        for key in max_keys:
            merged[key] = max(int(merged.get(key, 0)), int(new_diag.get(key, 0)))
        for key in float_keys:
            merged[key] = float(merged.get(key, 0.0)) + float(new_diag.get(key, 0.0))

        stop_counts = dict(merged.get("stop_reason_counts", {}))
        for reason, count in dict(new_diag.get("stop_reason_counts", {})).items():
            stop_counts[str(reason)] = int(stop_counts.get(str(reason), 0)) + int(count)
        merged["stop_reason_counts"] = stop_counts

        merged_slow_gen = list(merged.get("slowest_generate_calls", []))
        for item in list(new_diag.get("slowest_generate_calls", [])):
            if isinstance(item, dict):
                self._record_top_slowest(merged_slow_gen, dict(item), key="elapsed_s", keep_top_k=3)
        merged["slowest_generate_calls"] = merged_slow_gen

        merged_slow_search = list(merged.get("slowest_search_calls", []))
        for item in list(new_diag.get("slowest_search_calls", [])):
            if isinstance(item, dict):
                self._record_top_slowest(merged_slow_search, dict(item), key="elapsed_s", keep_top_k=3)
        merged["slowest_search_calls"] = merged_slow_search
        return merged

    def _cleanup_model_runner_graph_buffers(self, reason: str = "") -> None:
        if self.model_runner is None:
            return

        graph_runner = getattr(self.model_runner, "graph_runner", None)
        if graph_runner is None:
            return

        for attr in ("graphs", "output_buffers"):
            obj = getattr(graph_runner, attr, None)
            if hasattr(obj, "clear"):
                obj.clear()
            if hasattr(graph_runner, attr):
                setattr(graph_runner, attr, None)

        for attr in ("input_ids", "position_ids", "past_key_values", "attention_mask",
                     "_bounded_kv_views", "_shared_graph_hidden", "_last_kv_input"):
            if hasattr(graph_runner, attr):
                setattr(graph_runner, attr, None)

        self.model_runner.graph_runner = None

        try:
            from dinfer.decoding import diffusion_runner as diffusion_runner_mod

            if hasattr(diffusion_runner_mod, "set_global_graph_memory_pool"):
                diffusion_runner_mod.set_global_graph_memory_pool(None)
        except Exception:
            pass

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        suffix = f" ({reason})" if reason else ""
        print(f"[dInfer][rank{self.rank}] released model_runner graph buffers{suffix}", flush=True)

    def _build_runtime_modules(self) -> None:
        if self.model is None:
            raise RuntimeError("DInfer model is missing before runtime module build.")
        if self.device is None:
            raise RuntimeError("DInfer device is missing before runtime module build.")
        if self._server_args is None:
            raise RuntimeError("DInfer server args are missing before runtime module build.")
        if self._model_config is None:
            raise RuntimeError("DInfer model config is missing before runtime module build.")

        max_total_len = self.length + 2048
        prefill_lengths = ([32] if getattr(self, "full_prompt_prefill", False)
                           else [32, 64, 128, 256, 512, 1024, 2048, 4096])
        supported_batch_sizes = sorted({1, self.max_batch_size} | {
            2 ** i for i in range(self.max_batch_size.bit_length()) if 2 ** i <= self.max_batch_size})

        # Suppress verbose per-iteration diffusion logs and keep rollout logs readable.
        logging.getLogger("dinfer.decoding.generate_uniform").setLevel(logging.WARNING)

        # Work around a known dinfer logging-format bug during CUDA graph capture.
        if self.runtime_cfg.use_compile:
            logging.getLogger("dinfer.decoding.diffusion_runner").setLevel(logging.WARNING)

        from . import b200_moe
        b200_moe.install(self)
        runtime_deps = _get_model_runtime_deps()
        optimize_decode = bool(getattr(self, "optimize_decode", False))
        if optimize_decode:
            if self.parallel_mode != "dp":
                raise ValueError("Optimized editable decoding requires DP inference")
            from . import sparse_decode, decode_kv
            sparse_decode.prepare_head(self)
            decode_kv.prepare_attention(self)

        # Retain graph outputs in shared compact buffers. PyTorch slice_scatter
        # can otherwise retain an entire multi-layer backing storage per KV view.
        if optimize_decode:
            from .decode_graphs import bounded_graph_capture
            max_total_len = max(max_total_len, 8192)
            graph_context = bounded_graph_capture(self.max_batch_size * max_total_len)
        else:
            graph_context = nullcontext()
        from .cache_conditioning import masked_decode_graphs
        with masked_decode_graphs(), graph_context:
            self.model_runner = runtime_deps["ModelRunner"](
                self.model,
                self.device,
                server_args=self._server_args,
                max_length=max_total_len,
                prefill_lengths=prefill_lengths,
                enable_cuda_graph=self.runtime_cfg.use_compile,
                supported_batch_sizes=supported_batch_sizes,
                use_cross_block=(self.max_batch_size == 1),
            )

        # Upstream ModelRunner still allocates graph static buffers even when cuda-graph is disabled.
        if not self.runtime_cfg.use_compile:
            self._cleanup_model_runner_graph_buffers(reason="compile_disabled")

        mask_id = getattr(self._model_config, "mask_token_id", 156895)
        eos_id = getattr(self._model_config, "eos_token_id", 156892)
        if eos_id is None:
            eos_id = 156892
        self._mask_id = int(mask_id)
        self._eos_id = int(eos_id)

        self.decoder = runtime_deps["TraceRLThresholdParallelDecoder"](
            temperature=0.0,
            threshold=self.threshold,
            mask_id=self._mask_id,
            eos_id=self._eos_id,
            deterministic_zero_temp=(
                self.runtime_cfg.use_compile and self.runtime_cfg.deterministic_zero_temp
            ),
        )

        cache_factory = runtime_deps["KVCacheFactory"](
            "prefix",
            is_bd_model=True,
            backend="sglang",
            max_length=max_total_len,
        )

        self.dllm = runtime_deps["BlockDiffusionLLM"](
            self.model_runner,
            self.decoder,
            runtime_deps["BlockIteratorFactory"](start_block_align=True, use_block_diffusion=True),
            cache_factory=cache_factory,
            early_stop=True,
            maximum_unroll=4,
            expected_tpf=4,
            backend="sglang",
            mini_batch_size=self.max_batch_size,
            prefilling_limit=self.runtime_cfg.prefilling_limit,
            use_naive_batching=True,
        )

        if getattr(self, "full_prompt_prefill", False):
            self._install_full_prompt_prefill()

        if optimize_decode:
            from . import decode_kv
            sparse_decode.finish_install(self)
            decode_kv.install_replay_cache(self)
            decode_kv.install_block_updates(self)
            decode_kv.install_eager_storage(self)

        from .cache_conditioning import install_cache_conditioning
        install_cache_conditioning(self)

    def _install_full_prompt_prefill(self):
        """Prefill full history once, retaining block-causal attention and KV state.

        The upstream 128-token first chunk followed by 32-token forwards streams
        every model weight repeatedly for each long action prefix. Prefill never
        consumes vocabulary logits, so call the decoder without the LM head.
        """
        from types import MethodType
        engine = self
        self._rl_original_prefill = self.dllm.block_runner.prefill
        # Whole-prefix prefill uses the learner's real-token block mask.
        @torch.no_grad()
        def prefill(runner, model, prefilling_x, kv_cache, pos_ids, attn_mask,
                    prefilling_limit, block_length):
            if kv_cache is not None:
                _, past = engine.model.model(
                    prefilling_x, pos_ids, None, None, use_cache=True,
                    attention_mask=attn_mask,
                )
                kv_cache.range_update(past, 0, prefilling_x.size(1), 0)
                runner.diff_iteration.num_forwards += 1
                runner.diff_iteration.iter_no += 1
            runner.need_cross_block_update = False

        self.dllm.block_runner.prefill = MethodType(prefill, self.dllm.block_runner)

    def _teardown_runtime_modules(self, reason: str = "") -> None:
        self._cleanup_model_runner_graph_buffers(reason=reason)
        self.dllm = None
        self.decoder = None
        self.model_runner = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _initialize_model_parallel_allow_data_parallel(
        parallel_state_module,
        tensor_model_parallel_size: int,
        expert_model_parallel_size: int,
        pipeline_model_parallel_size: int,
        backend: str,
    ) -> None:
        """Initialize SGLang model-parallel groups while allowing data-parallel replicas.

        Upstream SGLang currently enforces world_size == tp * pp. We need to support
        replicated single-rank TP groups (tp=1, pp=1) under multi-process training.
        """
        assert torch.distributed.is_initialized()
        world_size = torch.distributed.get_world_size()

        model_parallel_world = tensor_model_parallel_size * pipeline_model_parallel_size
        if world_size % model_parallel_world != 0:
            raise RuntimeError(
                f"world_size ({world_size}) must be divisible by "
                f"tensor_model_parallel_size ({tensor_model_parallel_size}) x "
                f"pipeline_model_parallel_size ({pipeline_model_parallel_size})"
            )

        num_tensor_model_parallel_groups = world_size // tensor_model_parallel_size
        assert parallel_state_module._TP is None, "tensor model parallel group is already initialized"
        group_ranks = []
        for i in range(num_tensor_model_parallel_groups):
            ranks = list(
                range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size)
            )
            group_ranks.append(ranks)

        parallel_state_module._TP = parallel_state_module.init_model_parallel_group(
            group_ranks,
            parallel_state_module.get_world_group().local_rank,
            backend,
            use_message_queue_broadcaster=parallel_state_module.get_bool_env_var(
                "SGLANG_USE_MESSAGE_QUEUE_BROADCASTER", "true"
            ),
            group_name="tp",
        )

        moe_ep_size = expert_model_parallel_size
        moe_tp_size = tensor_model_parallel_size // moe_ep_size

        assert parallel_state_module._MOE_EP is None, "expert model parallel group is already initialized"
        if moe_ep_size == tensor_model_parallel_size:
            parallel_state_module._MOE_EP = parallel_state_module._TP
        else:
            group_ranks = []
            for i in range(num_tensor_model_parallel_groups):
                for j in range(moe_tp_size):
                    st = i * tensor_model_parallel_size + j
                    en = (i + 1) * tensor_model_parallel_size + j
                    ranks = list(range(st, en, moe_tp_size))
                    group_ranks.append(ranks)
            parallel_state_module._MOE_EP = parallel_state_module.init_model_parallel_group(
                group_ranks,
                parallel_state_module.get_world_group().local_rank,
                backend,
                group_name="moe_ep",
            )

        assert parallel_state_module._MOE_TP is None, "expert model parallel group is already initialized"
        if moe_tp_size == tensor_model_parallel_size:
            parallel_state_module._MOE_TP = parallel_state_module._TP
        else:
            group_ranks = []
            for i in range(num_tensor_model_parallel_groups):
                for j in range(moe_ep_size):
                    st = i * tensor_model_parallel_size + j * moe_tp_size
                    en = i * tensor_model_parallel_size + (j + 1) * moe_tp_size
                    ranks = list(range(st, en))
                    group_ranks.append(ranks)
            parallel_state_module._MOE_TP = parallel_state_module.init_model_parallel_group(
                group_ranks,
                parallel_state_module.get_world_group().local_rank,
                backend,
                group_name="moe_tp",
            )

        num_pipeline_model_parallel_groups = world_size // pipeline_model_parallel_size
        assert parallel_state_module._PP is None, "pipeline model parallel group is already initialized"
        group_ranks = []
        for i in range(num_pipeline_model_parallel_groups):
            ranks = list(range(i, world_size, num_pipeline_model_parallel_groups))
            group_ranks.append(ranks)
        parallel_state_module._PP = parallel_state_module.init_model_parallel_group(
            group_ranks,
            parallel_state_module.get_world_group().local_rank,
            backend,
            use_custom_allreduce=False,
            group_name="pp",
        )

    @staticmethod
    def _patch_tp_collectives() -> None:
        """Patch dInfer TP collectives to use the TP process group instead of WORLD."""
        import sglang.srt.distributed as sglang_distributed

        def safe_tp_all_reduce(tensor: torch.Tensor) -> torch.Tensor:
            tp_world_size = sglang_distributed.get_tensor_model_parallel_world_size()
            if tp_world_size == 1:
                return tensor
            tp_group = sglang_distributed.get_tp_group()
            torch.distributed.all_reduce(tensor, group=tp_group.device_group)
            return tensor

        def safe_tp_all_gather(input_: torch.Tensor) -> torch.Tensor:
            tp_world_size = sglang_distributed.get_tensor_model_parallel_world_size()
            if tp_world_size == 1:
                return input_
            input_size = input_.size()
            output_size = (input_size[0] * tp_world_size,) + input_size[1:]
            output_tensor = torch.empty(output_size, dtype=input_.dtype, device=input_.device)
            tp_group = sglang_distributed.get_tp_group()
            torch.distributed.all_gather_into_tensor(
                output_tensor, input_, group=tp_group.device_group
            )
            return output_tensor

        # Patch both distributed namespace and model module local symbol.
        sglang_distributed.tensor_model_parallel_all_reduce = safe_tp_all_reduce
        sglang_distributed.tensor_model_parallel_all_gather = safe_tp_all_gather

        # Do not import the dInfer model module just to patch it: importing
        # dinfer.model.modeling_llada2_moe_sglang has global side effects and
        # can perturb the following learner/FSDP training path. If the module is
        # already loaded, patch its captured local symbols; otherwise it will
        # import the safe functions from sglang_distributed later.
        llada2_mod = sys.modules.get("dinfer.model.modeling_llada2_moe_sglang")
        if llada2_mod is not None:
            llada2_mod.tensor_model_parallel_all_reduce = safe_tp_all_reduce
            llada2_mod.sglang_distributed.tensor_model_parallel_all_reduce = safe_tp_all_reduce
            llada2_mod.sglang_distributed.tensor_model_parallel_all_gather = safe_tp_all_gather

    @staticmethod
    def _patch_decoder_broadcast() -> None:
        """Patch decoder broadcast helper to avoid WORLD-group sync in DP mode."""
        import sglang.srt.distributed as sglang_distributed

        parallel_strategy = sys.modules.get("dinfer.decoding.parallel_strategy")
        if parallel_strategy is None:
            return

        def safe_broadcast_if_needed(x: torch.Tensor, src: int = 0, group=None) -> None:
            if not (dist.is_available() and dist.is_initialized()):
                return

            # Respect explicit group usage if provided by callers.
            if group is not None:
                if dist.get_world_size(group) > 1:
                    dist.broadcast(x, src=src, group=group)
                return

            tp_world_size = sglang_distributed.get_tensor_model_parallel_world_size()
            if tp_world_size <= 1:
                return

            tp_group = sglang_distributed.get_tp_group()
            dist.broadcast(x, src=src, group=tp_group.device_group)

        parallel_strategy.broadcast_if_needed = safe_broadcast_if_needed

    @staticmethod
    def _initialize_dp_attention_safe(server_args: Any, model_config: Any, parallel_mode: str) -> None:
        """Initialize DP-attention metadata without creating invalid groups in replicated DP mode."""
        import sglang.srt.layers.dp_attention as dp_attention_mod
        from sglang.srt.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
            get_tp_group,
        )

        if parallel_mode != "dp":
            dp_attention_mod.initialize_dp_attention(server_args=server_args, model_config=model_config)
            return

        # In replicated DP (tp_size=1 per replica), upstream initialize_dp_attention
        # assumes a single global tp index space and may create an invalid group map.
        # For our DP rollout path we only need local single-replica attention metadata.
        tp_group = get_tp_group()
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()

        dp_attention_mod._ATTN_TP_GROUP = tp_group
        dp_attention_mod._ATTN_TP_RANK = tp_rank
        dp_attention_mod._ATTN_TP_SIZE = tp_size
        dp_attention_mod._ATTN_DP_RANK = 0
        dp_attention_mod._ATTN_DP_SIZE = 1
        dp_attention_mod._LOCAL_ATTN_DP_RANK = 0
        dp_attention_mod._LOCAL_ATTN_DP_SIZE = 1
        dp_attention_mod._ENABLE_DP_ATTENTION_FLAG = False
        dp_attention_mod._DpGatheredBufferWrapper.set_metadata(
            hidden_size=model_config.hidden_size,
            dtype=model_config.dtype,
            device=torch.device(server_args.device),
        )

    @torch.no_grad()
    def initialize(self) -> None:
        if self.initialized:
            return

        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError("torch.distributed must be initialized before dInfer setup.")

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for dInfer rollout engine.")

        from sglang.srt import distributed

        self._distributed = distributed

        torch.cuda.set_device(self.local_rank)
        self.device = torch.device("cuda", self.local_rank)

        os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "127.0.0.1")
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")

        if self.parallel_mode == "tp":
            tensor_model_parallel_size = self.world_size
            expert_model_parallel_size = self.world_size
            server_tp_size = self.world_size
        else:
            tensor_model_parallel_size = 1
            expert_model_parallel_size = 1
            server_tp_size = 1

        distributed.init_distributed_environment(
            world_size=self.world_size,
            rank=self.rank,
            distributed_init_method="env://",
            local_rank=self.local_rank,
            backend="nccl",
            timeout=self.runtime_cfg.dist_timeout_sec,
        )
        if self.parallel_mode == "dp":
            import sglang.srt.distributed.parallel_state as parallel_state

            if not parallel_state.model_parallel_is_initialized():
                backend = torch.distributed.get_backend(parallel_state.get_world_group().device_group)
                self._initialize_model_parallel_allow_data_parallel(
                    parallel_state_module=parallel_state,
                    tensor_model_parallel_size=tensor_model_parallel_size,
                    expert_model_parallel_size=expert_model_parallel_size,
                    pipeline_model_parallel_size=1,
                    backend=backend,
                )
            else:
                distributed.ensure_model_parallel_initialized(
                    tensor_model_parallel_size=tensor_model_parallel_size,
                    expert_model_parallel_size=expert_model_parallel_size,
                    pipeline_model_parallel_size=1,
                    backend="nccl",
                )
        else:
            distributed.ensure_model_parallel_initialized(
                tensor_model_parallel_size=tensor_model_parallel_size,
                expert_model_parallel_size=expert_model_parallel_size,
                pipeline_model_parallel_size=1,
                backend="nccl",
            )
        self._patch_tp_collectives()
        self._patch_decoder_broadcast()

        print(
            f"[dInfer][rank{self.rank}] parallel_mode={self.parallel_mode} "
            f"(tp_size={server_tp_size}, world_size={self.world_size})",
            flush=True,
        )

        if self._provided_tokenizer is not None and self._provided_model_config is not None:
            print(
                f"[dInfer][rank{self.rank}] reusing tokenizer/config from learner process",
                flush=True,
            )
            self.tokenizer = self._provided_tokenizer
            self._model_config = self._provided_model_config
            dist.barrier()
        else:
            # Serialize HF tokenizer/config loading across ranks to avoid file-lock races.
            self.tokenizer = None
            self._model_config = None
            for src_rank in range(self.world_size):
                if self.rank == src_rank:
                    print(
                        f"[dInfer][rank{self.rank}] initializing tokenizer/config from {self.model_path}",
                        flush=True,
                    )
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.model_path,
                        trust_remote_code=True,
                        local_files_only=True,
                    )
                    self._model_config = AutoConfig.from_pretrained(
                        self.model_path,
                        trust_remote_code=True,
                        local_files_only=True,
                    )
                dist.barrier()

        if self.tokenizer is None or self._model_config is None:
            raise RuntimeError(f"[dInfer][rank{self.rank}] failed to initialize tokenizer/config.")

        enable_dp_attention = self.parallel_mode == "tp"
        self._server_args = _get_server_args_cls()(
            model_path=self.model_path,
            tokenizer_path=self.model_path,
            served_model_name=self.model_path,
            enable_dp_attention=enable_dp_attention,
            trust_remote_code=True,
            tp_size=server_tp_size,
            dp_size=1,
            pp_size=1,
            device=str(self.device),
            random_seed=0,
        )

        try:
            from sglang.srt.server_args import set_global_server_args_for_scheduler

            set_global_server_args_for_scheduler(self._server_args)
        except ImportError:
            pass

        print(f"[dInfer][rank{self.rank}] init dp-attention metadata...", flush=True)
        self._initialize_dp_attention_safe(
            server_args=self._server_args,
            model_config=self._model_config,
            parallel_mode=self.parallel_mode,
        )
        print(f"[dInfer][rank{self.rank}] dp-attention metadata ready", flush=True)

        print(f"[dInfer][rank{self.rank}] init moe config...", flush=True)
        _get_initialize_moe_config()(self._server_args)
        print(f"[dInfer][rank{self.rank}] moe config ready", flush=True)

        print(f"[dInfer][rank{self.rank}] loading model weights...", flush=True)
        model_runtime_deps = _get_model_runtime_deps()
        self.model = model_runtime_deps["LLaDA2SGLangLM"](
            config=self._model_config,
            expert_map_path=self.expert_map_path,
        ).eval()
        prev_default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            self.model.load_weights(self.model_path, device=self.device)
        finally:
            torch.set_default_dtype(prev_default_dtype)
        _get_initialize_moe_config()(self._server_args)
        self.model = self.model.to(self.device)
        # Initial load can leave routed-expert `.data` aliases resident on CUDA until we clear them.
        self._clear_routed_expert_aliases(reason="post_init_load")

        self._build_runtime_modules()

        if self.parallel_mode == "dp" and not self._dp_warmup_enabled:
            print(
                f"[dInfer][rank{self.rank}] warmup skipped in DP mode "
                "(TRACE_RL_DP_WARMUP=0)",
                flush=True,
            )
        else:
            self._run_generate_warmup(mode=self.parallel_mode)

        if self.parallel_mode == "dp" or self.rank == 0:
            self.search_client = SearchClient(url=self.retriever_url, topk=self.topk_search)

        dist.barrier()
        self.initialized = True
        self.weights_loaded = True
        print(f"[dInfer][rank{self.rank}] ready", flush=True)

    @torch.no_grad()
    def update_from_train_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        if not self.initialized:
            raise RuntimeError("DInferRolloutEngine must be initialized first.")
        if self.device is None:
            raise RuntimeError("DInferRolloutEngine device is not initialized.")
        if self.model is None:
            raise RuntimeError("DInferRolloutEngine model is not initialized.")

        if self.model.quant_config is not None:
            raise RuntimeError("Prepared rank-local sync payloads do not support quantized rollout models.")
        payload_bundle = self.build_sync_payload_for_rank(state_dict, target_rank=self.rank)
        self.apply_prepared_sync_payload(payload_bundle)

    def build_sync_payload_for_rank(
        self,
        state_dict: Dict[str, torch.Tensor],
        *,
        target_rank: int,
    ) -> Dict[str, Any]:
        model_config = self.model.config if self.model is not None else self._model_config
        if model_config is None:
            raise RuntimeError("Model config is not initialized for sync payload construction.")
        return trace_rl_prepare_rank_local_sync_payload(
            state_dict,
            model_config=model_config,
            expert_map_path=self.expert_map_path,
            target_rank=target_rank,
            target_world_size=self.world_size,
            parallel_mode=self.parallel_mode,
        )

    def build_sync_state_dict_for_rank(
        self,
        state_dict: Dict[str, torch.Tensor],
        *,
        target_rank: int,
    ) -> Dict[str, torch.Tensor]:
        payload_bundle = self.build_sync_payload_for_rank(state_dict, target_rank=target_rank)
        return dict(payload_bundle.get("weights", {}) or {})

    @torch.no_grad()
    def apply_prepared_sync_payload(self, payload_bundle: Dict[str, Any]) -> None:
        if not self.initialized:
            raise RuntimeError("DInferRolloutEngine must be initialized first.")
        if self.device is None:
            raise RuntimeError("DInferRolloutEngine device is not initialized.")
        if self.model is None:
            raise RuntimeError("DInferRolloutEngine model is not initialized.")

        self._move_model_to_device_if_needed()
        weights = dict(payload_bundle.get("weights", {}) or {})
        expert_map_cpu_by_layer = list(payload_bundle.get("expert_map_cpu_by_layer", []) or [])

        self._last_sync_used_rebind = False
        self._last_sync_guard_failures = []
        self._last_sync_requires_runtime_rebuild = False

        setattr(self.model, "_trace_rl_sync_inplace_mode", True)
        try:
            layer_owner = getattr(self.model, "model", self.model)
            model_layers = getattr(layer_owner, "layers", None)
            if model_layers is None:
                raise AttributeError(f"{type(self.model).__name__} does not expose model layers")
            for layer_id, expert_map_cpu in enumerate(expert_map_cpu_by_layer):
                if layer_id >= len(model_layers):
                    break
                layer = model_layers[layer_id]
                if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
                    layer.mlp.experts.expert_map_cpu = expert_map_cpu
            self.model.apply_state_dicts(weights)
        finally:
            setattr(self.model, "_trace_rl_sync_inplace_mode", False)

        self.model.init_h2e_module()
        if hasattr(self.model, "after_processing"):
            self.model.after_processing()
        self._clear_routed_expert_aliases(reason="post_sync")

        self._last_sync_used_rebind = bool(getattr(self.model, "_last_apply_used_rebind", False))
        self._last_sync_guard_failures = list(
            getattr(self.model, "_last_apply_inplace_guard_failures", []) or []
        )
        self._last_sync_requires_runtime_rebuild = bool(self._last_sync_used_rebind)

        if self._last_sync_used_rebind:
            print(
                f"[dInfer][rank{self.rank}] in-place sync guard fell back to storage rebinding; "
                "runtime rebuild will remain enabled for safety",
                flush=True,
            )

        if self.runtime_cfg.use_compile and self._last_sync_requires_runtime_rebuild and (
            self.model_runner is not None or self.decoder is not None or self.dllm is not None
        ):
            print(
                f"[dInfer][rank{self.rank}] compile enabled after sync fallback; "
                "dropping runtime modules to force fresh graph capture",
                flush=True,
            )
            self._teardown_runtime_modules(reason="post_sync_storage_rebind_fallback")

        self.weights_loaded = True
        dist.barrier()

    def should_teardown_runtime_before_sync(self) -> bool:
        return False

    @torch.no_grad()
    def reload_from_path(self, model_dir: str) -> None:
        if not self.initialized:
            raise RuntimeError("DInferRolloutEngine must be initialized first.")
        if self.device is None:
            raise RuntimeError("DInferRolloutEngine device is not initialized.")
        self.model.load_weights(model_dir, device=self.device)
        self.model.init_h2e_module()
        if hasattr(self.model, "after_processing"):
            self.model.after_processing()
        self._clear_routed_expert_aliases(reason="post_reload")
        if self.model_runner is not None or self.decoder is not None or self.dllm is not None:
            print(
                f"[dInfer][rank{self.rank}] disk reload completed; rebuilding runtime modules",
                flush=True,
            )
            self._teardown_runtime_modules(reason="post_reload_storage_rebind")
        self.weights_loaded = True
        dist.barrier()

    def _move_model_to_device_if_needed(self) -> None:
        if self.model is None or self.device is None:
            return

        try:
            first_param = next(self.model.parameters())
        except StopIteration:
            first_param = None

        if first_param is None:
            return

        if first_param.device != self.device:
            self.model = self.model.to(self.device)

    @torch.no_grad()
    def unload_weights_keep_engine(self) -> None:
        if not self.initialized:
            raise RuntimeError("DInferRolloutEngine must be initialized first.")

        if not self.weights_loaded:
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
            return

        if self.model is None:
            raise RuntimeError("DInferRolloutEngine model is missing.")

        print(f"[dInfer][rank{self.rank}] offloading dInfer weights to CPU (engine retained)", flush=True)
        self._log_local_cuda_mem("before_offload")
        self._clear_model_side_runtime_refs(reason="pre_offload")
        self.model = self.model.to("cpu")
        self.weights_loaded = False
        self._clear_model_side_runtime_refs(reason="post_offload")

        # CUDA graph buffers are tied to old parameter pointers; drop them after offload.
        self._cleanup_model_runner_graph_buffers(reason="post_offload")

        if self.full_runtime_unload:
            print(
                f"[dInfer][rank{self.rank}] full runtime teardown enabled; releasing runner/decoder caches",
                flush=True,
            )
            self._teardown_runtime_modules(reason="full_runtime_unload")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._log_local_cuda_mem("after_offload")

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    @torch.no_grad()
    def generate_rollouts(
        self,
        batch_items: List[Dict[str, Any]],
        k_sample: int,
        search_cfg: Any,
        rollout_cfg: Any,
    ) -> List[Dict[str, Any]]:
        self._last_rollout_perf = {
            "gen_elapsed_s": 0.0,
            "gen_tokens": 0.0,
            "diag": self._new_rollout_diag(),
        }

        if not self.initialized:
            raise RuntimeError("DInferRolloutEngine must be initialized first.")
        if not self.weights_loaded:
            raise RuntimeError("dInfer weights are unloaded. Sync/load weights before rollout.")
        if self.model_runner is None or self.decoder is None or self.dllm is None:
            print(f"[dInfer][rank{self.rank}] rebuilding runtime modules before rollout", flush=True)
            self._move_model_to_device_if_needed()
            self._build_runtime_modules()
            if dist.is_available() and dist.is_initialized():
                dist.barrier()

        max_retrieval_turns = int(
            search_cfg.get("max_retrieval_turns", search_cfg.get("max_turns", self.max_turns))
        )
        max_obs_length = int(search_cfg.get("max_obs_length", self.max_obs_length))
        topk = int(search_cfg.get("topk", self.topk_search))
        global_token_pool = int(rollout_cfg.get("global_max_token_pool", rollout_cfg.get("max_token", self.length)))
        max_token_per_call_multi = int(rollout_cfg.get("max_token_per_call_multi", 256))
        max_token_per_call_single = int(rollout_cfg.get("max_token_per_call_single", 256))
        max_no_search_tokens = int(rollout_cfg.get("max_no_search_tokens", 256))
        k_sample = max(1, int(k_sample))

        if self.rank == 0 and not self._rollout_limits_logged:
            print(
                "[dInfer] rollout_limits "
                f"global_token_pool={global_token_pool}, "
                f"max_retrieval_turns={max_retrieval_turns}, "
                f"max_token_per_call_multi={max_token_per_call_multi}, "
                f"max_token_per_call_single={max_token_per_call_single}, "
                f"max_no_search_tokens={max_no_search_tokens}",
                flush=True,
            )
            self._rollout_limits_logged = True

        if hasattr(self.decoder, "temperature"):
            self.decoder.temperature = float(rollout_cfg.get("temperature", 0.0))
        if hasattr(self.decoder, "threshold"):
            self.decoder.threshold = float(rollout_cfg.get("dynamic_threshold", self.threshold))

        if self.parallel_mode == "dp":
            return self._generate_rollouts_dp(
                batch_items=batch_items,
                k_sample=k_sample,
                max_retrieval_turns=max_retrieval_turns,
                max_obs_length=max_obs_length,
                topk=topk,
                global_token_pool=global_token_pool,
                max_token_per_call_multi=max_token_per_call_multi,
                max_token_per_call_single=max_token_per_call_single,
                max_no_search_tokens=max_no_search_tokens,
            )

        if self.rank == 0:
            base_prompts, base_ground_truths = self._prepare_rollout_inputs(batch_items)
        else:
            base_prompts = None
            base_ground_truths = None

        shared_obj = [base_prompts, base_ground_truths]
        dist.broadcast_object_list(shared_obj, src=0)
        base_prompts, base_ground_truths = shared_obj[0], shared_obj[1]

        per_sample_outputs = []
        total_gen_elapsed_s = 0.0
        total_gen_tokens = 0
        aggregated_diag = self._new_rollout_diag()
        for _ in range(k_sample):
            histories, step_maps, info_masks_list, sample_stats, perf = self._generate_with_search(
                prompts=base_prompts,
                max_retrieval_turns=max_retrieval_turns,
                max_obs_length=max_obs_length,
                topk=topk,
                global_token_pool=global_token_pool,
                max_token_per_call_multi=max_token_per_call_multi,
                max_token_per_call_single=max_token_per_call_single,
                max_no_search_tokens=max_no_search_tokens,
            )
            total_gen_elapsed_s += float(perf.get("gen_elapsed_s", 0.0))
            total_gen_tokens += int(perf.get("gen_tokens", 0))
            aggregated_diag = self._merge_rollout_diag(aggregated_diag, perf.get("diag"))

            if self.rank == 0:
                per_sample_outputs.append(
                    self._build_sample_results(
                        base_prompts=base_prompts,
                        base_ground_truths=base_ground_truths,
                        histories=histories,
                        step_maps=step_maps,
                        info_masks_list=info_masks_list,
                        sample_stats=sample_stats,
                    )
                )

        if self.rank == 0:
            results = self._merge_k_sample_results(per_sample_outputs=per_sample_outputs, k_sample=k_sample)
            self._last_rollout_perf = {
                "gen_elapsed_s": float(total_gen_elapsed_s),
                "gen_tokens": float(total_gen_tokens),
                "generated_tokens_model_only": float(total_gen_tokens),
                "diag": aggregated_diag,
            }
        else:
            results = None

        out_obj = [results]
        dist.broadcast_object_list(out_obj, src=0)
        return out_obj[0]

    def _prepare_rollout_inputs(self, batch_items: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        base_prompts = []
        base_ground_truths = []
        for item in batch_items:
            question = item.get("question", item.get("prompt", ""))
            prompt = _build_search_prompt(question)
            gt = item.get("answer", item.get("target", ""))
            base_prompts.append(prompt)
            base_ground_truths.append(gt)
        return base_prompts, base_ground_truths

    def _get_stop_token_sequences(self) -> List[List[int]]:
        if self._stop_token_sequences is None:
            self._stop_token_sequences = self._encode_stop_token_sequences(["</search>", "</answer>"])
        return self._stop_token_sequences

    def _encode_stop_token_sequences(self, stop_texts: Iterable[str]) -> List[List[int]]:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer must be initialized before stop-token encoding.")
        sequences: List[List[int]] = []
        for stop_text in stop_texts:
            # The tokenizer may merge leading whitespace with "</" (e.g.
            # " </search>" tokenizes differently from "</search>"). Include
            # common whitespace-prefixed variants so dInfer stops at the tag
            # instead of generating a full per-call chunk that Python later trims.
            for prefix in ("", " ", "\n", "\n\n", "\t"):
                ids = self.tokenizer.encode(prefix + str(stop_text), add_special_tokens=False)
                if ids:
                    sequences.append([int(tok) for tok in ids])
        deduped: List[List[int]] = []
        seen = set()
        for seq in sequences:
            key = tuple(seq)
            if key not in seen:
                seen.add(key)
                deduped.append(seq)
        return deduped

    def _ensure_ready_for_generate(self) -> None:
        if not self.initialized:
            raise RuntimeError("DInferRolloutEngine must be initialized first.")
        if not self.weights_loaded:
            raise RuntimeError("dInfer weights are unloaded. Sync/load weights before rollout.")
        if self.model_runner is None or self.decoder is None or self.dllm is None:
            print(f"[dInfer][rank{self.rank}] rebuilding runtime modules before generation", flush=True)
            self._move_model_to_device_if_needed()
            self._build_runtime_modules()
            if dist.is_available() and dist.is_initialized():
                dist.barrier()

    @torch.no_grad()
    def generate_text_once(
        self,
        *,
        prompt_text: str,
        cfg: Any,
        stop_texts: Iterable[str],
        banned_token_ids: Optional[Iterable[int]] = None,
        return_step_map: bool = False,
        return_token_ids: bool = False,
    ) -> Tuple[str, int, Dict[str, Any]]:
        """Generate an Action-Mode continuation with dInfer.

        With return_step_map, return generated token IDs and their denoising commit
        steps in info. Trace arrays are sliced in token space to preserve alignment.
        """

        self._ensure_ready_for_generate()
        if self.tokenizer is None or self.device is None:
            raise RuntimeError("dInfer tokenizer/device is not initialized.")

        max_new_tokens = max(1, int(getattr(cfg, "max_new_tokens", self.length)))
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_len = len(prompt_ids)
        input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        gen_len_aligned = _BUCKET_SIZE * ((prompt_len + max_new_tokens) // _BUCKET_SIZE) - prompt_len
        if gen_len_aligned < max_new_tokens:
            gen_len_aligned += _BUCKET_SIZE
        valid_gen_len = max(1, int(gen_len_aligned))

        old_temperature = getattr(self.decoder, "temperature", None)
        old_threshold = getattr(self.decoder, "threshold", None)
        old_banned = set(getattr(self.decoder, "banned_token_ids", set()) or set())
        if hasattr(self.decoder, "temperature"):
            self.decoder.temperature = float(getattr(cfg, "temperature", 0.0))
        if hasattr(self.decoder, "threshold"):
            self.decoder.threshold = float(getattr(cfg, "confidence_threshold", self.threshold))
        if hasattr(self.decoder, "set_banned_token_ids"):
            self.decoder.set_banned_token_ids(banned_token_ids)

        t_gen = time.time()
        try:
            out = self.dllm.generate(
                input_tensor,
                gen_length=valid_gen_len,
                block_length=max(1, int(getattr(cfg, "block_length", self.block_length))),
                return_step_map=bool(return_step_map),
                stop_token_sequences=self._encode_stop_token_sequences(stop_texts),
                stop_check_stride_blocks=1,
                max_new_tokens_per_seq=torch.tensor([max_new_tokens], dtype=torch.long, device=self.device),
            )
        finally:
            if old_temperature is not None and hasattr(self.decoder, "temperature"):
                self.decoder.temperature = old_temperature
            if old_threshold is not None and hasattr(self.decoder, "threshold"):
                self.decoder.threshold = old_threshold
            if hasattr(self.decoder, "set_banned_token_ids"):
                self.decoder.set_banned_token_ids(old_banned)

        seq_step_map = None
        if isinstance(out, tuple):
            out_tensor = out[0]
            if return_step_map and len(out) > 1 and out[1] is not None:
                # step_map is (batch, total_length) over the full prompt+gen canvas.
                seq_step_map = out[1][0]
        else:
            out_tensor = out
        # EOS-trim in TOKEN space, then slice the generated region [prompt_len:].
        new_ids = _cut_eos_1d(out_tensor[0, prompt_len:], self._eos_id)
        text = self.tokenizer.decode(new_ids, skip_special_tokens=False)
        token_count = len(self.tokenizer.encode(text, add_special_tokens=False)) if text else 0
        info = {
            "backend": "dinfer",
            "prompt_tokens": int(prompt_len),
            "generated_tokens": int(token_count),
            "requested_gen_len": int(max_new_tokens),
            "aligned_gen_len": int(valid_gen_len),
            "elapsed_s": float(time.time() - t_gen),
        }
        if return_token_ids:
            # Raw committed generated token ids (TOKEN space, EOS-trimmed), parallel to step_map.
            info["token_ids"] = [int(t) for t in new_ids.tolist()]
        if return_step_map:
            if seq_step_map is not None:
                gen_len = int(new_ids.shape[0])
                # Align the step_map slice to the generated region [prompt_len : prompt_len+gen_len].
                seg_step_map = seq_step_map[prompt_len : prompt_len + gen_len]
                info["step_map"] = [int(s) for s in seg_step_map.tolist()]
            else:
                info["step_map"] = []
        if return_step_map:
            self._attach_action_replay(info, prompt_ids, cfg, banned_token_ids, 0)
        return text, token_count, info

    def _attach_action_replay(self, info, prompt_ids, cfg, banned_token_ids, row):
        replay = getattr(self.dllm, "_last_action_replay", None)
        if replay is None:
            raise RuntimeError("dInfer did not capture complete terminal-block replay")
        start, end = len(prompt_ids), int(replay["end_positions"][row])
        info.update(
            trace_version=2,
            prompt_token_ids=list(prompt_ids),
            conditioning_token_ids=replay["token_ids"][row, start:end].tolist(),
            conditioning_step_map=replay["step_map"][row, start:end].tolist(),
            sampling_temperature=float(cfg.temperature),
            banned_token_ids=sorted(int(t) for t in (banned_token_ids or [])),
        )

    @torch.no_grad()
    def generate_text_batch_once(
        self,
        *,
        prompt_texts: List[str],
        cfg: Any,
        stop_texts: Iterable[str],
        banned_token_ids: Optional[Iterable[int]] = None,
        return_step_map: bool = False,
        return_token_ids: bool = False,
    ) -> List[Tuple[str, int, Dict[str, Any]]]:
        """Generate a ragged batch of append actions, preserving token-space traces.

        Right padding uses masks that the diffusion iterator fills from each
        sequence's original prompt boundary. Per-sequence budgets/terminal tags
        stop each rollout independently.
        """

        self._ensure_ready_for_generate()
        if self.tokenizer is None or self.device is None:
            raise RuntimeError("dInfer tokenizer/device is not initialized.")

        if not prompt_texts:
            return []
        if len(prompt_texts) > self.max_batch_size:
            raise ValueError("Action batch exceeds dInfer max_batch_size")
        max_new_tokens = max(1, int(getattr(cfg, "max_new_tokens", self.length)))
        encoded = [self.tokenizer.encode(text, add_special_tokens=False) for text in prompt_texts]
        prompt_lens = [len(ids) for ids in encoded]
        prompt_len = max(prompt_lens)
        input_tensor = torch.full((len(encoded), prompt_len), self.decoder.mask_id,
                                  dtype=torch.long, device=self.device)
        for i, ids in enumerate(encoded):
            input_tensor[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=self.device)
        valid_gen_len = _BUCKET_SIZE * ((prompt_len + max_new_tokens + _BUCKET_SIZE - 1) // _BUCKET_SIZE) - prompt_len

        old_temperature = getattr(self.decoder, "temperature", None)
        old_threshold = getattr(self.decoder, "threshold", None)
        old_banned = set(getattr(self.decoder, "banned_token_ids", set()) or set())
        if hasattr(self.decoder, "temperature"):
            self.decoder.temperature = float(getattr(cfg, "temperature", 0.0))
        if hasattr(self.decoder, "threshold"):
            self.decoder.threshold = float(getattr(cfg, "confidence_threshold", self.threshold))
        if hasattr(self.decoder, "set_banned_token_ids"):
            self.decoder.set_banned_token_ids(banned_token_ids)

        t_gen = time.time()
        try:
            out = self.dllm.generate(
                input_tensor,
                gen_length=valid_gen_len,
                block_length=max(1, int(getattr(cfg, "block_length", self.block_length))),
                return_step_map=bool(return_step_map),
                stop_token_sequences=self._encode_stop_token_sequences(stop_texts),
                stop_check_stride_blocks=1,
                max_new_tokens_per_seq=torch.full((len(encoded),), max_new_tokens, dtype=torch.long, device=self.device),
            )
        finally:
            if old_temperature is not None and hasattr(self.decoder, "temperature"):
                self.decoder.temperature = old_temperature
            if old_threshold is not None and hasattr(self.decoder, "threshold"):
                self.decoder.threshold = old_threshold
            if hasattr(self.decoder, "set_banned_token_ids"):
                self.decoder.set_banned_token_ids(old_banned)

        out_tensor, maps = (out[0], out[1]) if isinstance(out, tuple) else (out, None)
        # One host transfer for the complete batch, not one synchronization for
        # every token/sequence field. Raw token IDs remain aligned to step maps.
        out_cpu = out_tensor.cpu()
        maps_cpu = maps.cpu() if maps is not None else None
        elapsed = time.time() - t_gen
        results = []
        for i, prefix_length in enumerate(prompt_lens):
            new_ids = _cut_eos_1d(out_cpu[i, prefix_length:], self._eos_id)
            text = self.tokenizer.decode(new_ids, skip_special_tokens=False)
            count = len(self.tokenizer.encode(text, add_special_tokens=False)) if text else 0
            info = {"backend": "dinfer", "prompt_tokens": prefix_length,
                    "generated_tokens": count, "requested_gen_len": max_new_tokens,
                    "aligned_gen_len": valid_gen_len, "elapsed_s": elapsed,
                    "action_batch_size": len(encoded)}
            if return_token_ids:
                info["token_ids"] = new_ids.tolist()
            if return_step_map:
                info["step_map"] = (maps_cpu[i, prefix_length:prefix_length + len(new_ids)].tolist()
                                    if maps_cpu is not None else [])
            if return_step_map:
                self._attach_action_replay(info, encoded[i], cfg, banned_token_ids, i)
            results.append((text, count, info))
        return results

    @torch.no_grad()
    def forward_edit_logits_batch(self, requests):
        """Full bidirectional edit canvases, projecting only active mask slots.

        Right padding is excluded from attention; original token positions and
        the entire prefix/plan/evidence suffix remain visible for each episode.
        No KV cache is reused across an in-place plan edit.
        """
        if self.parallel_mode != "dp":
            raise ValueError("Batched editable canvases require DP inference")
        lengths = [len(request["canvas_ids"]) for request in requests]
        width = max(lengths)
        ids = torch.full((len(requests), width), self._eos_id,
                         dtype=torch.long, device=self.device)
        for i, request in enumerate(requests):
            ids[i, :lengths[i]] = torch.tensor(request["canvas_ids"], device=self.device)
        pos = torch.arange(width, device=self.device).expand(len(requests), -1)
        real = pos < torch.tensor(lengths, device=self.device)[:, None]
        attention = real[:, :, None] & real[:, None, :]
        diagonal = torch.arange(width, device=self.device)
        attention[:, diagonal, diagonal] = True
        hidden, _ = self.model.model(ids, pos, None, None, use_cache=False,
                                     attention_mask=attention)
        rows = [i for i, request in enumerate(requests) for _ in request["positions"]]
        cols = [position for request in requests for position in request["positions"]]
        selected = hidden[torch.tensor(rows, device=self.device), torch.tensor(cols, device=self.device)]
        logits = self.model.lm_head(selected)
        result = list(logits.split([len(request["positions"]) for request in requests]))
        if all(request.get("return_choices", False) for request in requests):
            from ..sampling import sample_edit_choices_batch
            return sample_edit_choices_batch(result, requests)
        return result

    def _append_information_obs(
        self,
        histories: List[str],
        step_maps: List[List[int]],
        info_masks_list: List[List[Tuple[int, int]]],
        base_prompt_lens: List[int],
        sample_idx: int,
        content: str,
    ) -> None:
        if content == _RETRIEVAL_LIMIT_FAILURE:
            info_obs = (
                "<information>\n"
                f"{content} Answer directly now.\n"
                "</information>"
            )
        else:
            info_obs = f"<information>\n{content}\n</information>\n"

        # Match SFT serialization: retrieval observations are user/environment
        # turns, not assistant-generated text. This keeps the next model action
        # aligned with the training pattern: information -> fresh assistant turn.
        obs_prefix = "<|role_end|><role>HUMAN</role>"
        obs_suffix = "<|role_end|><role>ASSISTANT</role>"
        obs = f"{obs_prefix}{info_obs}{obs_suffix}"
        obs_start_resp = len(histories[sample_idx]) - base_prompt_lens[sample_idx]
        info_start_resp = max(0, obs_start_resp + len(obs_prefix))
        info_end_resp = info_start_resp + len(info_obs)
        info_masks_list[sample_idx].append((info_start_resp, info_end_resp))

        histories[sample_idx] += obs
        obs_tok_len = len(self.tokenizer.encode(obs, add_special_tokens=False))
        step_maps[sample_idx].extend([-1] * obs_tok_len)

    def _build_rollout_sample_stats(
        self,
        generated_tokens_model_only: List[int],
        remaining_token_pool: List[int],
        retrieval_used: List[int],
        stop_reasons: List[str],
    ) -> List[Dict[str, Any]]:
        stats: List[Dict[str, Any]] = []
        for idx in range(len(generated_tokens_model_only)):
            stats.append(
                {
                    "generated_tokens_model_only": int(generated_tokens_model_only[idx]),
                    "remaining_token_pool": int(remaining_token_pool[idx]),
                    "retrieval_used": int(retrieval_used[idx]),
                    "stop_reason": stop_reasons[idx] or "unknown",
                }
            )
        return stats

    def _build_sample_results(
        self,
        base_prompts: List[str],
        base_ground_truths: List[str],
        histories: List[str],
        step_maps: List[List[int]],
        info_masks_list: List[List[Tuple[int, int]]],
        sample_stats: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        sample_results = []
        for idx, response_text in enumerate(histories):
            extracted = _extract_answer(response_text)
            response_len = len(self.tokenizer.encode(response_text, add_special_tokens=False))
            result = {
                "prompt": base_prompts[idx],
                "full_output": response_text,
                "step_map": step_maps[idx],
                "info_masks": info_masks_list[idx],
                "extracted_output": extracted or "",
                "response_length": response_len,
                "ground_truth": base_ground_truths[idx],
            }
            if sample_stats is not None and idx < len(sample_stats):
                result.update(sample_stats[idx])
            sample_results.append(result)
        return sample_results

    def _merge_k_sample_results(
        self, per_sample_outputs: List[List[Dict[str, Any]]], k_sample: int
    ) -> List[Dict[str, Any]]:
        if not per_sample_outputs:
            return []
        results = []
        for item_idx in range(len(per_sample_outputs[0])):
            for sample_idx in range(k_sample):
                results.append(per_sample_outputs[sample_idx][item_idx])
        return results

    def _finalize_response_step_maps(
        self,
        prompts: List[str],
        histories: List[str],
        step_maps: List[List[int]],
    ) -> Tuple[List[str], List[List[int]]]:
        response_texts = []
        final_step_maps = []
        for i, text in enumerate(histories):
            response_text = text[len(prompts[i]) :]
            response_texts.append(response_text)

            token_len = len(self.tokenizer.encode(response_text, add_special_tokens=False))
            sm = step_maps[i]
            if len(sm) > token_len:
                sm = sm[:token_len]
            elif len(sm) < token_len:
                non_neg = [x for x in sm if x >= 0]
                fill_val = (max(non_neg) + 1) if non_neg else 0
                sm = sm + [fill_val] * (token_len - len(sm))
            sm = _force_mask_environment_obs_tokens(response_text, sm, self.tokenizer)
            final_step_maps.append(sm)
        return response_texts, final_step_maps

    @torch.no_grad()
    def _generate_rollouts_dp(
        self,
        batch_items: List[Dict[str, Any]],
        k_sample: int,
        max_retrieval_turns: int,
        max_obs_length: int,
        topk: int,
        global_token_pool: int,
        max_token_per_call_multi: int,
        max_token_per_call_single: int,
        max_no_search_tokens: int,
    ) -> List[Dict[str, Any]]:
        if not batch_items:
            return []
        if self.search_client is None:
            self.search_client = SearchClient(url=self.retriever_url, topk=self.topk_search)

        base_prompts, base_ground_truths = self._prepare_rollout_inputs(batch_items)
        per_sample_outputs = []
        total_gen_elapsed_s = 0.0
        total_gen_tokens = 0
        aggregated_diag = self._new_rollout_diag()
        for _ in range(k_sample):
            histories, step_maps, info_masks_list, sample_stats, perf = self._generate_with_search_local(
                prompts=base_prompts,
                max_retrieval_turns=max_retrieval_turns,
                max_obs_length=max_obs_length,
                topk=topk,
                global_token_pool=global_token_pool,
                max_token_per_call_multi=max_token_per_call_multi,
                max_token_per_call_single=max_token_per_call_single,
                max_no_search_tokens=max_no_search_tokens,
            )
            total_gen_elapsed_s += float(perf.get("gen_elapsed_s", 0.0))
            total_gen_tokens += int(perf.get("gen_tokens", 0))
            aggregated_diag = self._merge_rollout_diag(aggregated_diag, perf.get("diag"))
            per_sample_outputs.append(
                self._build_sample_results(
                    base_prompts=base_prompts,
                    base_ground_truths=base_ground_truths,
                    histories=histories,
                    step_maps=step_maps,
                    info_masks_list=info_masks_list,
                    sample_stats=sample_stats,
                )
            )
        self._last_rollout_perf = {
            "gen_elapsed_s": float(total_gen_elapsed_s),
            "gen_tokens": float(total_gen_tokens),
            "generated_tokens_model_only": float(total_gen_tokens),
            "diag": aggregated_diag,
        }
        return self._merge_k_sample_results(per_sample_outputs=per_sample_outputs, k_sample=k_sample)

    @torch.no_grad()
    def _generate_with_search(
        self,
        prompts: List[str],
        max_retrieval_turns: int,
        max_obs_length: int,
        topk: int,
        global_token_pool: int,
        max_token_per_call_multi: int,
        max_token_per_call_single: int,
        max_no_search_tokens: int,
    ) -> Tuple[
        List[str],
        List[List[int]],
        List[List[Tuple[int, int]]],
        List[Dict[str, Any]],
        Dict[str, Any],
    ]:
        batch_size_total = len(prompts)
        total_gen_elapsed_s = 0.0
        total_gen_tokens = 0
        diag = self._new_rollout_diag()

        if self.rank == 0:
            histories = [p for p in prompts]
            finished_mask = [False] * batch_size_total
            step_maps = [[] for _ in range(batch_size_total)]
            info_masks_list = [[] for _ in range(batch_size_total)]
            base_prompt_lens = [len(p) for p in prompts]
            remaining_token_pool = [max(0, int(global_token_pool)) for _ in range(batch_size_total)]
            retrieval_used = [0 for _ in range(batch_size_total)]
            no_search_tokens = [0 for _ in range(batch_size_total)]
            generated_tokens_model_only = [0 for _ in range(batch_size_total)]
            stop_reasons = ["" for _ in range(batch_size_total)]
            tag_parse_carry = ["" for _ in range(batch_size_total)]
            retrieval_limit_notifications = [0 for _ in range(batch_size_total)]
        else:
            histories = None
            finished_mask = None
            step_maps = None
            info_masks_list = None
            base_prompt_lens = None
            remaining_token_pool = None
            retrieval_used = None
            no_search_tokens = None
            generated_tokens_model_only = None
            stop_reasons = None
            tag_parse_carry = None
            retrieval_limit_notifications = None

        stop_token_sequences = self._get_stop_token_sequences()
        loop_iter = 0
        while True:
            loop_iter += 1
            if self.rank == 0:
                for idx in range(batch_size_total):
                    if finished_mask[idx]:
                        continue
                    if (
                        max_no_search_tokens > 0
                        and no_search_tokens[idx] >= max_no_search_tokens
                    ):
                        finished_mask[idx] = True
                        if not stop_reasons[idx]:
                            stop_reasons[idx] = "no_search_hard_cap"
                        continue
                    if remaining_token_pool[idx] <= 0:
                        finished_mask[idx] = True
                        if not stop_reasons[idx]:
                            stop_reasons[idx] = "pool_exhaust"

                active_indices = [idx for idx, finished in enumerate(finished_mask) if not finished]
                done = len(active_indices) == 0
            else:
                active_indices = None
                done = None

            done_obj = [done]
            dist.broadcast_object_list(done_obj, src=0)
            if done_obj[0]:
                break

            if self.rank == 0:
                input_ids_list = []
                per_seq_token_limits = []
                for idx in active_indices:
                    ids = self.tokenizer.encode(histories[idx], add_special_tokens=False)
                    input_ids_list.append(torch.tensor(ids, dtype=torch.long))
                    per_seq_limit = max(0, int(remaining_token_pool[idx]))
                    if max_no_search_tokens > 0:
                        no_search_left = max(0, int(max_no_search_tokens - no_search_tokens[idx]))
                        per_seq_limit = min(per_seq_limit, no_search_left)
                    per_seq_token_limits.append(per_seq_limit)
                payload = [active_indices, input_ids_list, per_seq_token_limits]
            else:
                payload = [None, None, None]

            dist.broadcast_object_list(payload, src=0)
            active_indices, input_ids_list, per_seq_token_limits = payload[0], payload[1], payload[2]

            max_len = max(int(t.shape[0]) for t in input_ids_list)
            batch_curr = len(input_ids_list)
            diag["loop_iters"] += 1
            diag["generate_calls"] += 1
            diag["active_batch_total"] += int(batch_curr)
            diag["active_batch_max"] = max(int(diag["active_batch_max"]), int(batch_curr))
            if batch_curr == 1:
                diag["active_singleton_iters"] += 1
            else:
                diag["active_multi_iters"] += 1
            diag["prompt_max_total"] += int(max_len)
            diag["prompt_max_max"] = max(int(diag["prompt_max_max"]), int(max_len))
            input_tensor = torch.full(
                (batch_curr, max_len),
                self._mask_id,
                dtype=torch.long,
                device=self.device,
            )

            prompt_lens = []
            for idx, t in enumerate(input_ids_list):
                l = int(t.shape[0])
                prompt_lens.append(l)
                input_tensor[idx, :l] = t.to(self.device)

            requested_gen_len = max(1, int(max(per_seq_token_limits)))
            if batch_curr > 1 and max_token_per_call_multi > 0:
                requested_gen_len = min(requested_gen_len, int(max_token_per_call_multi))
            elif batch_curr == 1 and max_token_per_call_single > 0:
                requested_gen_len = min(requested_gen_len, int(max_token_per_call_single))
            gen_len_aligned = _BUCKET_SIZE * ((max_len + requested_gen_len) // _BUCKET_SIZE) - max_len
            if gen_len_aligned < requested_gen_len:
                gen_len_aligned += _BUCKET_SIZE
            valid_gen_len = int(gen_len_aligned)
            diag["requested_gen_total"] += int(requested_gen_len)
            diag["requested_gen_max"] = max(int(diag["requested_gen_max"]), int(requested_gen_len))
            diag["aligned_gen_total"] += int(valid_gen_len)
            diag["aligned_gen_max"] = max(int(diag["aligned_gen_max"]), int(valid_gen_len))

            max_new_tokens_per_seq = torch.tensor(
                [max(0, int(v)) for v in per_seq_token_limits],
                dtype=torch.long,
                device=self.device,
            )

            self._log_generate_call(
                phase="start",
                path="dp_sync",
                loop_iter=loop_iter,
                batch_curr=batch_curr,
                max_len=max_len,
                requested_gen_len=requested_gen_len,
                valid_gen_len=valid_gen_len,
                per_seq_token_limits=[int(v) for v in per_seq_token_limits],
            )
            t_gen = time.time()
            out, turn_step_map = self.dllm.generate(
                input_tensor,
                gen_length=valid_gen_len,
                block_length=self.block_length,
                return_step_map=True,
                stop_token_sequences=stop_token_sequences,
                stop_check_stride_blocks=1,
                max_new_tokens_per_seq=max_new_tokens_per_seq,
            )
            gen_call_elapsed = time.time() - t_gen
            total_gen_elapsed_s += gen_call_elapsed
            min_limit = int(min(per_seq_token_limits)) if per_seq_token_limits else 0
            max_limit = int(max(per_seq_token_limits)) if per_seq_token_limits else 0
            self._record_top_slowest(
                diag["slowest_generate_calls"],
                {
                    "iter": int(loop_iter),
                    "elapsed_s": float(gen_call_elapsed),
                    "batch": int(batch_curr),
                    "prompt_max": int(max_len),
                    "requested_gen": int(requested_gen_len),
                    "aligned_gen": int(valid_gen_len),
                    "per_seq_limit_min": int(min_limit),
                    "per_seq_limit_max": int(max_limit),
                },
                key="elapsed_s",
                keep_top_k=3,
            )
            self._log_generate_call(
                phase="done",
                path="dp_sync",
                loop_iter=loop_iter,
                batch_curr=batch_curr,
                max_len=max_len,
                requested_gen_len=requested_gen_len,
                valid_gen_len=valid_gen_len,
                per_seq_token_limits=[int(v) for v in per_seq_token_limits],
                elapsed_s=gen_call_elapsed,
            )

            rank0_error = None
            if self.rank == 0:
                try:
                    search_queries = []
                    search_indices = []

                    for j, batch_idx in enumerate(active_indices):
                        full_ids = _cut_eos_1d(out[j], self._eos_id)
                        new_ids = full_ids[prompt_lens[j] :]
                        new_ids = _cut_eos_1d(new_ids, self._eos_id)
                        raw_gen_token_count = int(new_ids.shape[0])

                        new_text = self.tokenizer.decode(new_ids, skip_special_tokens=False)
                        new_text = _sanitize_generated_text(
                            new_text,
                            carry=tag_parse_carry[batch_idx],
                        )
                        gen_token_count = len(
                            self.tokenizer.encode(new_text, add_special_tokens=False)
                        ) if new_text else 0
                        total_gen_tokens += gen_token_count
                        generated_tokens_model_only[batch_idx] += gen_token_count
                        remaining_token_pool[batch_idx] = max(
                            0,
                            int(remaining_token_pool[batch_idx]) - gen_token_count,
                        )

                        if gen_token_count > 0:
                            aligned_slice = _aligned_step_slice(
                                turn_step_map_row=turn_step_map[j],
                                start=prompt_lens[j],
                                raw_gen_count=raw_gen_token_count,
                                final_text=new_text,
                                tokenizer=self.tokenizer,
                            )
                            step_maps[batch_idx].extend(aligned_slice)
                        elif raw_gen_token_count == 0 and not stop_reasons[batch_idx]:
                            stop_reasons[batch_idx] = "eos"

                        histories[batch_idx] += new_text

                        tag_parse_carry[batch_idx], search_q, answer_cand = _extract_new_control_tags(
                            carry=tag_parse_carry[batch_idx],
                            new_text=new_text,
                        )
                        if raw_gen_token_count > 0 and not new_text:
                            answer_cand = _MALFORMED_ANSWER_CLOSE_SENTINEL
                        self._log_tag_parse_event(
                            path="dp_sync",
                            loop_iter=loop_iter,
                            batch_idx=batch_idx,
                            gen_token_count=gen_token_count,
                            new_text=new_text,
                            search_q=search_q,
                            answer_cand=answer_cand,
                            retrieval_used=retrieval_used[batch_idx],
                            remaining_token_pool=remaining_token_pool[batch_idx],
                            no_search_tokens=no_search_tokens[batch_idx],
                            stop_reason=stop_reasons[batch_idx],
                        )

                        if answer_cand:
                            finished_mask[batch_idx] = True
                            if not stop_reasons[batch_idx]:
                                if answer_cand == _MALFORMED_ANSWER_CLOSE_SENTINEL:
                                    stop_reasons[batch_idx] = "answer_format_error"
                                    diag["answer_format_error_events"] += 1
                                    diag["answer_close_only_events"] += 1
                                else:
                                    stop_reasons[batch_idx] = "answer"
                        elif search_q:
                            diag["search_tag_events"] += 1
                            if retrieval_used[batch_idx] < max_retrieval_turns:
                                # Reset no-search budget only when retrieval actually executes.
                                no_search_tokens[batch_idx] = 0
                                retrieval_used[batch_idx] += 1
                                search_queries.append(search_q)
                                search_indices.append(batch_idx)
                            else:
                                # Give the model two answer-directly observations after the cap.
                                # A third search attempt is a hard format failure; otherwise a
                                # short repeated <search> loop can explode generate_calls.
                                diag["retrieval_limit_events"] += 1
                                if retrieval_limit_notifications[batch_idx] >= 2:
                                    finished_mask[batch_idx] = True
                                    if not stop_reasons[batch_idx]:
                                        stop_reasons[batch_idx] = "answer_format_error"
                                        diag["answer_format_error_events"] += 1
                                else:
                                    self._append_information_obs(
                                        histories=histories,
                                        step_maps=step_maps,
                                        info_masks_list=info_masks_list,
                                        base_prompt_lens=base_prompt_lens,
                                        sample_idx=batch_idx,
                                        content=_RETRIEVAL_LIMIT_FAILURE,
                                    )
                                    retrieval_limit_notifications[batch_idx] += 1
                                    no_search_tokens[batch_idx] = 0
                        elif gen_token_count > 0:
                            no_search_tokens[batch_idx] += gen_token_count
                        elif gen_token_count == 0:
                            finished_mask[batch_idx] = True
                            if not stop_reasons[batch_idx]:
                                stop_reasons[batch_idx] = "eos"

                        if (
                            max_no_search_tokens > 0
                            and no_search_tokens[batch_idx] >= max_no_search_tokens
                            and not finished_mask[batch_idx]
                        ):
                            finished_mask[batch_idx] = True
                            if not stop_reasons[batch_idx]:
                                stop_reasons[batch_idx] = "no_search_hard_cap"

                        if remaining_token_pool[batch_idx] <= 0 and not finished_mask[batch_idx]:
                            finished_mask[batch_idx] = True
                            if not stop_reasons[batch_idx]:
                                stop_reasons[batch_idx] = "pool_exhaust"

                    if search_queries:
                        t_search = time.time()
                        results = self.search_client.batch_search(search_queries, topk=topk)
                        search_elapsed = time.time() - t_search
                        diag["search_calls"] += 1
                        diag["search_queries_total"] += int(len(search_queries))
                        diag["search_batch_max"] = max(
                            int(diag["search_batch_max"]), int(len(search_queries))
                        )
                        diag["search_elapsed_s"] += float(search_elapsed)
                        self._record_top_slowest(
                            diag["slowest_search_calls"],
                            {
                                "iter": int(loop_iter),
                                "elapsed_s": float(search_elapsed),
                                "query_count": int(len(search_queries)),
                            },
                            key="elapsed_s",
                            keep_top_k=3,
                        )
                        for s_idx, res in zip(search_indices, results):
                            res_tokens = self.tokenizer.encode(res, add_special_tokens=False)
                            if len(res_tokens) > max_obs_length:
                                res = self.tokenizer.decode(
                                    res_tokens[:max_obs_length], skip_special_tokens=False
                                )
                            self._append_information_obs(
                                histories=histories,
                                step_maps=step_maps,
                                info_masks_list=info_masks_list,
                                base_prompt_lens=base_prompt_lens,
                                sample_idx=s_idx,
                                content=res,
                            )
                except Exception as e:
                    rank0_error = (
                        f"[dInfer][rank0] iter={loop_iter} postprocess failed: "
                        f"{type(e).__name__}: {e}"
                    )
                    print(rank0_error, flush=True)

            error_obj = [rank0_error]
            dist.broadcast_object_list(error_obj, src=0)
            if error_obj[0] is not None:
                raise RuntimeError(error_obj[0])

            dist.barrier()

        final_error = None
        if self.rank == 0:
            try:
                response_texts, final_step_maps = self._finalize_response_step_maps(
                    prompts=prompts,
                    histories=histories,
                    step_maps=step_maps,
                )
                sample_stats = self._build_rollout_sample_stats(
                    generated_tokens_model_only=generated_tokens_model_only,
                    remaining_token_pool=remaining_token_pool,
                    retrieval_used=retrieval_used,
                    stop_reasons=stop_reasons,
                )
                stop_counts: Dict[str, int] = {}
                for reason in stop_reasons:
                    key = reason or "unknown"
                    stop_counts[key] = int(stop_counts.get(key, 0)) + 1
                diag["stop_reason_counts"] = stop_counts
                diag["max_generated_tokens_per_sample"] = int(
                    max(generated_tokens_model_only) if generated_tokens_model_only else 0
                )
                diag["max_retrieval_used_per_sample"] = int(
                    max(retrieval_used) if retrieval_used else 0
                )
            except Exception as e:
                final_error = (
                    f"[dInfer][rank0] final postprocess failed: "
                    f"{type(e).__name__}: {e}"
                )
                print(final_error, flush=True)
                response_texts = None
                final_step_maps = None
                info_masks_list = None
                sample_stats = None
        else:
            response_texts = None
            final_step_maps = None
            info_masks_list = None
            sample_stats = None

        final_error_obj = [final_error]
        dist.broadcast_object_list(final_error_obj, src=0)
        if final_error_obj[0] is not None:
            raise RuntimeError(final_error_obj[0])

        out_obj = [response_texts, final_step_maps, info_masks_list, sample_stats]
        dist.broadcast_object_list(out_obj, src=0)
        return out_obj[0], out_obj[1], out_obj[2], out_obj[3], {
            "gen_elapsed_s": float(total_gen_elapsed_s),
            "gen_tokens": float(total_gen_tokens),
            "generated_tokens_model_only": float(total_gen_tokens),
            "diag": diag,
        }

    @torch.no_grad()
    def _generate_with_search_local(
        self,
        prompts: List[str],
        max_retrieval_turns: int,
        max_obs_length: int,
        topk: int,
        global_token_pool: int,
        max_token_per_call_multi: int,
        max_token_per_call_single: int,
        max_no_search_tokens: int,
    ) -> Tuple[
        List[str],
        List[List[int]],
        List[List[Tuple[int, int]]],
        List[Dict[str, Any]],
        Dict[str, Any],
    ]:
        batch_size_total = len(prompts)
        histories = [p for p in prompts]
        finished_mask = [False] * batch_size_total
        step_maps = [[] for _ in range(batch_size_total)]
        info_masks_list = [[] for _ in range(batch_size_total)]
        base_prompt_lens = [len(p) for p in prompts]
        total_gen_elapsed_s = 0.0
        total_gen_tokens = 0
        diag = self._new_rollout_diag()
        remaining_token_pool = [max(0, int(global_token_pool)) for _ in range(batch_size_total)]
        retrieval_used = [0 for _ in range(batch_size_total)]
        no_search_tokens = [0 for _ in range(batch_size_total)]
        generated_tokens_model_only = [0 for _ in range(batch_size_total)]
        stop_reasons = ["" for _ in range(batch_size_total)]
        tag_parse_carry = ["" for _ in range(batch_size_total)]
        retrieval_limit_notifications = [0 for _ in range(batch_size_total)]

        stop_token_sequences = self._get_stop_token_sequences()
        loop_iter = 0
        while True:
            loop_iter += 1
            for idx in range(batch_size_total):
                if finished_mask[idx]:
                    continue
                if (
                    max_no_search_tokens > 0
                    and no_search_tokens[idx] >= max_no_search_tokens
                ):
                    finished_mask[idx] = True
                    if not stop_reasons[idx]:
                        stop_reasons[idx] = "no_search_hard_cap"
                    continue
                if remaining_token_pool[idx] <= 0:
                    finished_mask[idx] = True
                    if not stop_reasons[idx]:
                        stop_reasons[idx] = "pool_exhaust"

            active_indices = [idx for idx, finished in enumerate(finished_mask) if not finished]
            if not active_indices:
                break

            input_ids_list = []
            per_seq_token_limits = []
            for idx in active_indices:
                ids = self.tokenizer.encode(histories[idx], add_special_tokens=False)
                input_ids_list.append(torch.tensor(ids, dtype=torch.long))
                per_seq_limit = max(0, int(remaining_token_pool[idx]))
                if max_no_search_tokens > 0:
                    no_search_left = max(0, int(max_no_search_tokens - no_search_tokens[idx]))
                    per_seq_limit = min(per_seq_limit, no_search_left)
                per_seq_token_limits.append(per_seq_limit)

            max_len = max(int(t.shape[0]) for t in input_ids_list)
            batch_curr = len(input_ids_list)
            diag["loop_iters"] += 1
            diag["generate_calls"] += 1
            diag["active_batch_total"] += int(batch_curr)
            diag["active_batch_max"] = max(int(diag["active_batch_max"]), int(batch_curr))
            if batch_curr == 1:
                diag["active_singleton_iters"] += 1
            else:
                diag["active_multi_iters"] += 1
            diag["prompt_max_total"] += int(max_len)
            diag["prompt_max_max"] = max(int(diag["prompt_max_max"]), int(max_len))
            input_tensor = torch.full(
                (batch_curr, max_len),
                self._mask_id,
                dtype=torch.long,
                device=self.device,
            )

            prompt_lens = []
            for idx, t in enumerate(input_ids_list):
                l = int(t.shape[0])
                prompt_lens.append(l)
                input_tensor[idx, :l] = t.to(self.device)

            requested_gen_len = max(1, int(max(per_seq_token_limits)))
            if batch_curr > 1 and max_token_per_call_multi > 0:
                requested_gen_len = min(requested_gen_len, int(max_token_per_call_multi))
            elif batch_curr == 1 and max_token_per_call_single > 0:
                requested_gen_len = min(requested_gen_len, int(max_token_per_call_single))
            gen_len_aligned = _BUCKET_SIZE * ((max_len + requested_gen_len) // _BUCKET_SIZE) - max_len
            if gen_len_aligned < requested_gen_len:
                gen_len_aligned += _BUCKET_SIZE
            valid_gen_len = int(gen_len_aligned)
            diag["requested_gen_total"] += int(requested_gen_len)
            diag["requested_gen_max"] = max(int(diag["requested_gen_max"]), int(requested_gen_len))
            diag["aligned_gen_total"] += int(valid_gen_len)
            diag["aligned_gen_max"] = max(int(diag["aligned_gen_max"]), int(valid_gen_len))

            max_new_tokens_per_seq = torch.tensor(
                [max(0, int(v)) for v in per_seq_token_limits],
                dtype=torch.long,
                device=self.device,
            )

            self._log_generate_call(
                phase="start",
                path="dp_local",
                loop_iter=loop_iter,
                batch_curr=batch_curr,
                max_len=max_len,
                requested_gen_len=requested_gen_len,
                valid_gen_len=valid_gen_len,
                per_seq_token_limits=[int(v) for v in per_seq_token_limits],
            )
            t_gen = time.time()
            out, turn_step_map = self.dllm.generate(
                input_tensor,
                gen_length=valid_gen_len,
                block_length=self.block_length,
                return_step_map=True,
                stop_token_sequences=stop_token_sequences,
                stop_check_stride_blocks=1,
                max_new_tokens_per_seq=max_new_tokens_per_seq,
            )
            gen_call_elapsed = time.time() - t_gen
            total_gen_elapsed_s += gen_call_elapsed
            min_limit = int(min(per_seq_token_limits)) if per_seq_token_limits else 0
            max_limit = int(max(per_seq_token_limits)) if per_seq_token_limits else 0
            self._record_top_slowest(
                diag["slowest_generate_calls"],
                {
                    "iter": int(loop_iter),
                    "elapsed_s": float(gen_call_elapsed),
                    "batch": int(batch_curr),
                    "prompt_max": int(max_len),
                    "requested_gen": int(requested_gen_len),
                    "aligned_gen": int(valid_gen_len),
                    "per_seq_limit_min": int(min_limit),
                    "per_seq_limit_max": int(max_limit),
                },
                key="elapsed_s",
                keep_top_k=3,
            )
            self._log_generate_call(
                phase="done",
                path="dp_local",
                loop_iter=loop_iter,
                batch_curr=batch_curr,
                max_len=max_len,
                requested_gen_len=requested_gen_len,
                valid_gen_len=valid_gen_len,
                per_seq_token_limits=[int(v) for v in per_seq_token_limits],
                elapsed_s=gen_call_elapsed,
            )

            search_queries = []
            search_indices = []

            for j, batch_idx in enumerate(active_indices):
                full_ids = _cut_eos_1d(out[j], self._eos_id)
                new_ids = full_ids[prompt_lens[j] :]
                new_ids = _cut_eos_1d(new_ids, self._eos_id)
                raw_gen_token_count = int(new_ids.shape[0])

                new_text = self.tokenizer.decode(new_ids, skip_special_tokens=False)
                new_text = _sanitize_generated_text(
                    new_text,
                    carry=tag_parse_carry[batch_idx],
                )
                gen_token_count = len(
                    self.tokenizer.encode(new_text, add_special_tokens=False)
                ) if new_text else 0
                total_gen_tokens += gen_token_count
                generated_tokens_model_only[batch_idx] += gen_token_count
                remaining_token_pool[batch_idx] = max(0, int(remaining_token_pool[batch_idx]) - gen_token_count)

                if gen_token_count > 0:
                    aligned_slice = _aligned_step_slice(
                        turn_step_map_row=turn_step_map[j],
                        start=prompt_lens[j],
                        raw_gen_count=raw_gen_token_count,
                        final_text=new_text,
                        tokenizer=self.tokenizer,
                    )
                    step_maps[batch_idx].extend(aligned_slice)
                elif raw_gen_token_count == 0 and not stop_reasons[batch_idx]:
                    stop_reasons[batch_idx] = "eos"

                histories[batch_idx] += new_text

                tag_parse_carry[batch_idx], search_q, answer_cand = _extract_new_control_tags(
                    carry=tag_parse_carry[batch_idx],
                    new_text=new_text,
                )
                if raw_gen_token_count > 0 and not new_text:
                    answer_cand = _MALFORMED_ANSWER_CLOSE_SENTINEL
                self._log_tag_parse_event(
                    path="dp_local",
                    loop_iter=loop_iter,
                    batch_idx=batch_idx,
                    gen_token_count=gen_token_count,
                    new_text=new_text,
                    search_q=search_q,
                    answer_cand=answer_cand,
                    retrieval_used=retrieval_used[batch_idx],
                    remaining_token_pool=remaining_token_pool[batch_idx],
                    no_search_tokens=no_search_tokens[batch_idx],
                    stop_reason=stop_reasons[batch_idx],
                )

                if answer_cand:
                    finished_mask[batch_idx] = True
                    if not stop_reasons[batch_idx]:
                        if answer_cand == _MALFORMED_ANSWER_CLOSE_SENTINEL:
                            stop_reasons[batch_idx] = "answer_format_error"
                            diag["answer_format_error_events"] += 1
                            diag["answer_close_only_events"] += 1
                        else:
                            stop_reasons[batch_idx] = "answer"
                elif search_q:
                    diag["search_tag_events"] += 1
                    if retrieval_used[batch_idx] < max_retrieval_turns:
                        # Reset no-search budget only when retrieval actually executes.
                        no_search_tokens[batch_idx] = 0
                        retrieval_used[batch_idx] += 1
                        search_queries.append(search_q)
                        search_indices.append(batch_idx)
                    else:
                        # Give the model two answer-directly observations after the cap.
                        # A third search attempt is a hard format failure; otherwise a
                        # short repeated <search> loop can explode generate_calls.
                        diag["retrieval_limit_events"] += 1
                        if retrieval_limit_notifications[batch_idx] >= 2:
                            finished_mask[batch_idx] = True
                            if not stop_reasons[batch_idx]:
                                stop_reasons[batch_idx] = "answer_format_error"
                                diag["answer_format_error_events"] += 1
                        else:
                            self._append_information_obs(
                                histories=histories,
                                step_maps=step_maps,
                                info_masks_list=info_masks_list,
                                base_prompt_lens=base_prompt_lens,
                                sample_idx=batch_idx,
                                content=_RETRIEVAL_LIMIT_FAILURE,
                            )
                            retrieval_limit_notifications[batch_idx] += 1
                            no_search_tokens[batch_idx] = 0
                elif gen_token_count > 0:
                    no_search_tokens[batch_idx] += gen_token_count
                elif gen_token_count == 0:
                    finished_mask[batch_idx] = True
                    if not stop_reasons[batch_idx]:
                        stop_reasons[batch_idx] = "eos"

                if (
                    max_no_search_tokens > 0
                    and no_search_tokens[batch_idx] >= max_no_search_tokens
                    and not finished_mask[batch_idx]
                ):
                    finished_mask[batch_idx] = True
                    if not stop_reasons[batch_idx]:
                        stop_reasons[batch_idx] = "no_search_hard_cap"

                if remaining_token_pool[batch_idx] <= 0 and not finished_mask[batch_idx]:
                    finished_mask[batch_idx] = True
                    if not stop_reasons[batch_idx]:
                        stop_reasons[batch_idx] = "pool_exhaust"

            if search_queries:
                t_search = time.time()
                results = self.search_client.batch_search(search_queries, topk=topk)
                search_elapsed = time.time() - t_search
                diag["search_calls"] += 1
                diag["search_queries_total"] += int(len(search_queries))
                diag["search_batch_max"] = max(
                    int(diag["search_batch_max"]), int(len(search_queries))
                )
                diag["search_elapsed_s"] += float(search_elapsed)
                self._record_top_slowest(
                    diag["slowest_search_calls"],
                    {
                        "iter": int(loop_iter),
                        "elapsed_s": float(search_elapsed),
                        "query_count": int(len(search_queries)),
                    },
                    key="elapsed_s",
                    keep_top_k=3,
                )
                for s_idx, res in zip(search_indices, results):
                    res_tokens = self.tokenizer.encode(res, add_special_tokens=False)
                    if len(res_tokens) > max_obs_length:
                        res = self.tokenizer.decode(
                            res_tokens[:max_obs_length], skip_special_tokens=False
                        )
                    self._append_information_obs(
                        histories=histories,
                        step_maps=step_maps,
                        info_masks_list=info_masks_list,
                        base_prompt_lens=base_prompt_lens,
                        sample_idx=s_idx,
                        content=res,
                    )

        response_texts, final_step_maps = self._finalize_response_step_maps(
            prompts=prompts,
            histories=histories,
            step_maps=step_maps,
        )
        sample_stats = self._build_rollout_sample_stats(
            generated_tokens_model_only=generated_tokens_model_only,
            remaining_token_pool=remaining_token_pool,
            retrieval_used=retrieval_used,
            stop_reasons=stop_reasons,
        )
        stop_counts: Dict[str, int] = {}
        for reason in stop_reasons:
            key = reason or "unknown"
            stop_counts[key] = int(stop_counts.get(key, 0)) + 1
        diag["stop_reason_counts"] = stop_counts
        diag["max_generated_tokens_per_sample"] = int(
            max(generated_tokens_model_only) if generated_tokens_model_only else 0
        )
        diag["max_retrieval_used_per_sample"] = int(
            max(retrieval_used) if retrieval_used else 0
        )

        return response_texts, final_step_maps, info_masks_list, sample_stats, {
            "gen_elapsed_s": float(total_gen_elapsed_s),
            "gen_tokens": float(total_gen_tokens),
            "generated_tokens_model_only": float(total_gen_tokens),
            "diag": diag,
        }

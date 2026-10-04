import json
import torch
from typing import Any, Dict, List, Optional, Union, Tuple


DIFFUPLAN_SYSTEM_PROMPT = (
    "You are an editable-plan search assistant. Follow the controller protocol exactly."
)

DIFFUPLAN_USER_RULES = "\n".join(
    [
        "You answer by maintaining and editing an explicit plan.",
        "Use only these assistant action tags: <think>, <plan>, <search>, <update>, <replan>, and <answer>.",
        "Use block formatting for every action:",
        "<tag>",
        "content",
        "</tag>",
        "Start by creating a plan in a <plan> block.",
        "Use a <think> block when you need to reason about the next action.",
        "Plans are ordered concrete subquestions, one per line: #Q_1:, #Q_2:, #Q_3:, ...",
        "Later plan steps may refer to earlier unknown answers as #A_1, #A_2, ..., but never future answers.",
        "Search results are environment context inside <information>...</information>; never generate <information> yourself.",
        "Do not output helper lines like #A_1: answer.",
        "When evidence gives the value of a placeholder used in the plan, emit <update> with #A_i markers only.",
        "After <update>, the controller turns the first matching #A_i in the plan into one editable mask slot.",
        "Use <replan> when the current plan suffix is wrong or needs a different path.",
        "During controller-forced plan denoising only, <|reserved_token_6|> means <expand>.",
        "<expand> is only valid inside the editable plan region during update/replan denoising.",
        "When enough evidence has been gathered and the plan is resolved, output a concise <answer> block.",
    ]
)

def process_mdm_sft_example(
    example: Dict[str, Any],
    tokenizer,
    max_seq_len: int,
    text_keys: Union[str, List[str]] = "messages",
    noise_range: Tuple[float, float] = (0.3, 0.8),
    mask_token_id: int = 156895,
    source_name: Optional[str] = None,
) -> List[Dict[str, "torch.Tensor"]]:
    if isinstance(text_keys, str):
        messages = example[text_keys]
    elif isinstance(text_keys, list):
        for key in text_keys:
            if key in example:
                messages = example[key]
                break
        else:
            raise ValueError(f"None of the keys {text_keys} are found in the example.")
    else:
        raise ValueError(f"text_keys must be a string or a list of strings, but got {type(text_keys)}")
    
    examples = []
    input_ids, prompt_length = apply_chat_template_mdm(messages=messages, tokenizer=tokenizer, max_length=max_seq_len)

    labels = input_ids.clone()
    labels[:prompt_length] = -100

    maskable_mask = torch.arange(max_seq_len) >= prompt_length

    noisy_input_ids = sft_noise_transition(
        input_ids.clone(), 
        noise_range=noise_range, 
        maskable_mask=maskable_mask, 
        mask_token_id=mask_token_id
    )

    loss_mask = noisy_input_ids == mask_token_id
    labels[~loss_mask] = -100

    examples.append({
        "input_ids": input_ids,
        "noisy_input_ids": noisy_input_ids,
        "attention_mask": torch.ones_like(input_ids), # 使用torch.ones_like更简洁
        "labels": labels,
    })
    
    return examples


def process_mdm_tokenized_example(
    example: Dict[str, List[int]],
    max_seq_len: int, 
    text_keys: Union[str, List[str]] = "input_ids",
    noise_range: Tuple[float, float] = (0.3, 0.8),
    mask_token_id: int = 156895,
    source_name: Optional[str] = None,
) -> List[Dict[str, "torch.Tensor"]]:
    examples = []
    if isinstance(text_keys, str):
        input_ids = example[text_keys]
    elif isinstance(text_keys, list):
        for text_key in text_keys:
            if text_key in example:
                input_ids = example[text_key]
                break
        else:
            raise ValueError(f"None of the keys {text_keys} are found in the example.")
    else:
        raise ValueError(f"text_keys must be a string or a list of strings, but got {type(text_keys)}")

    prompt_length = example['prompt_lengths']

    input_ids = torch.tensor(input_ids)
    labels = input_ids.clone()
    labels[:prompt_length] = -100

    maskable_mask = torch.arange(max_seq_len) >= prompt_length

    noisy_input_ids = sft_noise_transition(
        input_ids.clone(), 
        noise_range=noise_range, 
        maskable_mask=maskable_mask, 
        mask_token_id=mask_token_id
    )

    loss_mask = noisy_input_ids == mask_token_id
    labels[~loss_mask] = -100

    examples.append({
        "input_ids": input_ids,
        "noisy_input_ids": noisy_input_ids,
        "attention_mask": torch.ones_like(input_ids), # 使用torch.ones_like更简洁
        "labels": labels,
    })
    
    return examples


def process_diffuplan_sft_example(
    example: Dict[str, Any],
    tokenizer,
    max_seq_len: int,
    noise_range: Tuple[float, float] = (0.3, 0.8),
    mask_token_id: int = 156895,
    source_name: Optional[str] = None,
) -> List[Dict[str, "torch.Tensor"]]:
    record_type = example.get("record_type")
    if record_type == "edit":
        return [
            _process_diffuplan_edit_record(
                example,
                max_seq_len=max_seq_len,
                tokenizer=tokenizer,
                noise_range=noise_range,
                mask_token_id=mask_token_id,
            )
        ]
    if record_type == "action":
        return [
            _process_diffuplan_action_record(
                example,
                tokenizer=tokenizer,
                max_seq_len=max_seq_len,
                noise_range=noise_range,
                mask_token_id=mask_token_id,
            )
        ]
    raise ValueError(f"Unsupported Diffuplan record_type: {record_type}")


def _process_diffuplan_edit_record(
    example: Dict[str, Any],
    max_seq_len: int,
    tokenizer,
    noise_range: Tuple[float, float],
    mask_token_id: int,
) -> Dict[str, "torch.Tensor"]:
    pad_token_id = _get_pad_token_id(tokenizer)
    seq_len = min(len(example["input_token_ids"]), max_seq_len)
    input_ids = _pad_or_truncate(example["label_token_ids"], max_seq_len, pad_token_id)
    noisy_input_ids = _pad_or_truncate(example["input_token_ids"], max_seq_len, pad_token_id)
    edit_loss_mask = _pad_or_truncate(example["loss_mask"], max_seq_len, 0).bool()

    labels = input_ids.clone()
    labels[~edit_loss_mask] = -100

    edit_fields = _build_diffuplan_edit_fields(
        example=example,
        max_seq_len=max_seq_len,
        edit_loss_mask=edit_loss_mask,
    )

    return {
        "input_ids": input_ids,
        "noisy_input_ids": noisy_input_ids,
        "attention_mask": (torch.arange(max_seq_len) < seq_len).long(),
        "labels": labels,
        **edit_fields,
    }


def _process_diffuplan_action_record(
    example: Dict[str, Any],
    tokenizer,
    max_seq_len: int,
    noise_range: Tuple[float, float],
    mask_token_id: int,
) -> Dict[str, "torch.Tensor"]:
    pad_token_id = _get_pad_token_id(tokenizer)
    prompt_text = _format_diffuplan_action_prompt(example)
    target_text = example["target_text"]

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]
    if not target_ids:
        raise ValueError("Diffuplan action record has an empty target_text.")

    if len(target_ids) >= max_seq_len:
        prompt_ids = []
        target_ids = target_ids[:max_seq_len]
    else:
        prompt_budget = max_seq_len - len(target_ids)
        prompt_ids = prompt_ids[-prompt_budget:]

    prompt_len = len(prompt_ids)
    clean_unpadded = prompt_ids + target_ids
    seq_len = len(clean_unpadded)

    input_ids = _pad_or_truncate(clean_unpadded, max_seq_len, pad_token_id)
    labels = input_ids.clone()
    real_token_mask = torch.arange(max_seq_len) < seq_len
    maskable_mask = (torch.arange(max_seq_len) >= prompt_len) & real_token_mask

    noisy_input_ids = sft_noise_transition(
        input_ids.clone(),
        noise_range=noise_range,
        maskable_mask=maskable_mask,
        mask_token_id=mask_token_id,
    )
    if not ((noisy_input_ids == mask_token_id) & maskable_mask).any():
        forced_pos = torch.randint(prompt_len, seq_len, (1,)).item()
        noisy_input_ids[forced_pos] = mask_token_id

    loss_mask = (noisy_input_ids == mask_token_id) & maskable_mask
    labels[~loss_mask] = -100

    return {
        "input_ids": input_ids,
        "noisy_input_ids": noisy_input_ids,
        "attention_mask": real_token_mask.long(),
        "labels": labels,
        **_empty_diffuplan_edit_fields(max_seq_len),
    }


def _empty_diffuplan_edit_fields(max_seq_len: int) -> Dict[str, "torch.Tensor"]:
    return {
        "diffuplan_record_type": torch.tensor(0, dtype=torch.long),
        "diffuplan_edit_loss_mask": torch.zeros(max_seq_len, dtype=torch.bool),
        "diffuplan_edit_target_ids": torch.empty(0, dtype=torch.long),
        "diffuplan_edit_target_len": torch.tensor(0, dtype=torch.long),
        "diffuplan_edit_segment_kinds": torch.empty(0, dtype=torch.long),
        "diffuplan_edit_segment_lens": torch.empty(0, dtype=torch.long),
    }


def _build_diffuplan_edit_fields(
    *,
    example: Dict[str, Any],
    max_seq_len: int,
    edit_loss_mask: "torch.Tensor",
) -> Dict[str, "torch.Tensor"]:
    target_ids = [int(item) for item in (example.get("target_token_ids") or [])]
    target_ids = target_ids[:max_seq_len]
    segments = _loads_json_field(example.get("segments"), default=[])
    segment_kinds: List[int] = []
    segment_lens: List[int] = []
    if isinstance(segments, list):
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            segment_target_ids = segment.get("target_token_ids") or []
            if not segment_target_ids:
                continue
            state = str(segment.get("state") or "")
            segment_kinds.append(1 if state == "mask" else 0)
            segment_lens.append(len(segment_target_ids))

    return {
        "diffuplan_record_type": torch.tensor(1, dtype=torch.long),
        "diffuplan_edit_loss_mask": edit_loss_mask,
        "diffuplan_edit_target_ids": torch.tensor(target_ids, dtype=torch.long),
        "diffuplan_edit_target_len": torch.tensor(len(target_ids), dtype=torch.long),
        "diffuplan_edit_segment_kinds": torch.tensor(segment_kinds, dtype=torch.long),
        "diffuplan_edit_segment_lens": torch.tensor(segment_lens, dtype=torch.long),
    }


def _format_diffuplan_action_prompt(example: Dict[str, Any]) -> str:
    prompt_text = example.get("prompt_text")
    if prompt_text:
        return str(prompt_text)

    context = _loads_json_field(example.get("context", {}), default={})
    context_str = _render_diffuplan_context(context)
    action_type = example.get("action_type", "next")
    question = example.get("question", context.get("question", ""))
    return (
        f"<role>SYSTEM</role>{DIFFUPLAN_SYSTEM_PROMPT}<|role_end|>"
        f"<role>HUMAN</role>{DIFFUPLAN_USER_RULES}\n\n"
        f"Question: {question}\n\n"
        f"{context_str}\n\n"
        f"Produce the next {action_type} action."
        f"<|role_end|><role>ASSISTANT</role>"
    )


def _render_diffuplan_context(context: Dict[str, Any]) -> str:
    lines = ["Current state:"]
    old_plan = context.get("old_plan_text") or context.get("current_plan_text")
    if old_plan:
        lines.extend(["<plan>", str(old_plan).strip(), "</plan>"])

    solved = (
        context.get("solved_qas")
        or (context.get("context_before_edit") or {}).get("solved_qas")
        or []
    )
    if solved:
        lines.append("Solved answers:")
        for item in solved:
            step_id = item.get("id")
            answer = str(item.get("answer") or "").strip()
            if step_id and answer:
                lines.append(f"#A_{step_id}: {answer}")

    previous_searches = context.get("previous_searches") or []
    if previous_searches:
        lines.append("Previous searches:")
        for item in previous_searches[-3:]:
            query = str(item.get("query") or "").strip()
            if query:
                lines.append(f"- {query}")

    latest_information = str(context.get("latest_information") or "").strip()
    if latest_information:
        lines.extend(["Latest information:", "<information>", latest_information, "</information>"])

    failed_step = (context.get("context_before_edit") or {}).get("failed_step")
    if failed_step:
        lines.append("Failed step:")
        lines.append(json.dumps(failed_step, ensure_ascii=False, sort_keys=True))

    if len(lines) == 1:
        lines.append(json.dumps(context, ensure_ascii=False, sort_keys=True))
    return "\n".join(lines)


def _loads_json_field(value: Any, default: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return value if value is not None else default


def _get_pad_token_id(tokenizer) -> int:
    if getattr(tokenizer, "pad_token_id", None) is not None:
        return tokenizer.pad_token_id
    if getattr(tokenizer, "eos_token_id", None) is not None:
        return tokenizer.eos_token_id
    return 0


def _pad_or_truncate(values: List[int], max_seq_len: int, pad_value: int) -> "torch.Tensor":
    values = list(values[:max_seq_len])
    if len(values) < max_seq_len:
        values = values + [pad_value] * (max_seq_len - len(values))
    return torch.tensor(values, dtype=torch.long)


def sft_noise_transition(x_0, noise_range, maskable_mask, mask_token_id):
    """
    Performs a noise transition by masking tokens.

    Args:
        x_0 (torch.Tensor): The input sequence (batch_size, seq_len).
        noise_range (tuple): A tuple (min, max) for the noise range, from which the masking ratio sigma is sampled.
        maskable_mask (torch.Tensor): A boolean mask indicating which positions are allowed to be masked (batch_size, seq_len).
        mask_token_id (int): The ID of the mask token.

    Returns:
        torch.Tensor: The sequence after masking.
    """

    t_tensor = torch.rand(1) * (noise_range[1] - noise_range[0]) + noise_range[0]
    sigma = t_tensor.item()
    # move_chance = 1 - (-sigma).exp()
    move_chance = sigma
    move_indices = (torch.rand(*x_0.shape) < move_chance) & maskable_mask
    x_t = torch.where(move_indices, mask_token_id, x_0)
    return x_t

def apply_chat_template_mdm(messages, tokenizer, max_length):
    inputs_str = tokenizer.apply_chat_template(messages, tokenize=False)
    prompt_str = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)

    prompt_ids_unpadded = tokenizer(prompt_str, add_special_tokens=False)['input_ids']
    prompt_length = len(prompt_ids_unpadded)

    tokenized_input = tokenizer(
        inputs_str,
        return_tensors="pt",
        truncation=True, 
        max_length=max_length, 
        padding="max_length",
        add_special_tokens=False
    ).input_ids.squeeze(0)

    return tokenized_input, prompt_length

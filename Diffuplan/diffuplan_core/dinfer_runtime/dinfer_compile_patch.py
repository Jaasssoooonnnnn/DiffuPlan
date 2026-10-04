import os

import torch


def _get_sparse_moe_block_cls():
    from dinfer.model.modeling_llada2_moe_sglang import LLaDA2SparseMoeBlock
    return LLaDA2SparseMoeBlock


def _get_custom_op_cls():
    from sglang.srt.custom_op import CustomOp
    return CustomOp


def _get_monkey_patch_torch_compile():
    from sglang.srt.utils.patch_torch import monkey_patch_torch_compile
    return monkey_patch_torch_compile


def apply_compile_exclusion_patches() -> None:
    LLaDA2SparseMoeBlock = _get_sparse_moe_block_cls()
    if getattr(LLaDA2SparseMoeBlock, "_compile_exclusion_patched", False):
        return

    # Run sparse MoE eagerly within the compiled model.
    LLaDA2SparseMoeBlock.forward_normal = torch.compiler.disable(
        LLaDA2SparseMoeBlock.forward_normal
    )
    LLaDA2SparseMoeBlock._compile_exclusion_patched = True


def patch_torch_compile_config() -> None:
    if os.environ.get("DINFER_DISABLE_COORDINATE_DESCENT_TUNING", "0") != "1":
        return
    if getattr(patch_torch_compile_config, "_patched", False):
        return

    import torch._dynamo.config
    import torch._inductor.config
    import dinfer.decoding.diffusion_runner as dinfer_diffusion_runner

    monkey_patch_torch_compile = _get_monkey_patch_torch_compile()

    def _apply_config():
        torch._inductor.config.coordinate_descent_tuning = False
        # Coordinate descent only controls tuning refinement. The initial
        # pointwise/persistent-reduction search otherwise benchmarks each GPU
        # independently and can select different floating-point reduction trees.
        torch._inductor.config.triton.autotune_pointwise = False
        torch._inductor.config.triton.unique_kernel_names = True
        torch._inductor.config.fx_graph_cache = True
        torch._dynamo.config.accumulated_cache_size_limit = 1024
        if hasattr(torch._dynamo.config, "cache_size_limit"):
            torch._dynamo.config.cache_size_limit = 1024
        monkey_patch_torch_compile()

    dinfer_diffusion_runner.set_torch_compile_config = _apply_config
    _apply_config()
    patch_torch_compile_config._patched = True


def patch_custom_op_logging() -> None:
    CustomOp = _get_custom_op_cls()
    preserve = {
        name.strip()
        for name in os.environ.get("DINFER_PRESERVE_CUSTOMOPS_IN_COMPILE", "").split(",")
        if name.strip()
    }
    should_log = os.environ.get("DINFER_LOG_ENTER_TORCH_COMPILE", "0") == "1"
    if not should_log and not preserve:
        return
    if getattr(CustomOp, "_dinfer_enter_logging_patched", False):
        return

    original_enter = CustomOp.enter_torch_compile

    def wrapped_enter(self, num_tokens: int):
        before = getattr(self._forward_method, "__name__", type(self._forward_method).__name__)
        if self.__class__.__name__ in preserve and not self.is_torch_compile:
            self._original_forward_method = self._forward_method
            self.is_torch_compile = True
        else:
            original_enter(self, num_tokens)
        after = getattr(self._forward_method, "__name__", type(self._forward_method).__name__)
        if should_log and (before != after or os.environ.get("DINFER_LOG_ALL_CUSTOMOPS", "0") == "1"):
            print(
                f"[enter_torch_compile] class={self.__class__.__name__} "
                f"num_tokens={num_tokens} before={before} after={after}",
                flush=True,
            )

    CustomOp.enter_torch_compile = wrapped_enter
    CustomOp._dinfer_enter_logging_patched = True

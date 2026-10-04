"""Stream FSDP1 parameters directly into resident, unquantized DP dInfer weights."""
from collections import defaultdict
import re
import time

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


def _clean(name):
    return '.'.join(p for p in name.split('.') if p not in {'_fsdp_wrapped_module', '_orig_mod'})


def _managed_tensors(module, prefix=''):
    # A summon_full_params(recurse=False) context owns only this FSDP unit.
    for name, value in module.named_parameters(recurse=False):
        yield prefix + name, value
    for name, value in module.named_buffers(recurse=False):
        if name not in module._non_persistent_buffers_set:
            yield prefix + name, value
    for name, child in module.named_children():
        if not isinstance(child, FSDP):
            yield from _managed_tensors(child, prefix + name + '.')


def _destinations(name, source, targets):
    if name.endswith('.mlp.w1') or name.endswith('.mlp.w2'):
        suffix = '.experts.w13_weight' if name.endswith('.w1') else '.experts.w2_weight'
        key = name.rsplit('.', 1)[0] + suffix
        return [(key, targets[key])]
    if '.mlp.experts.' in name:
        match = re.fullmatch(r'(.*\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight', name)
        if not match:
            raise ValueError(f'Unsupported expert parameter: {name}')
        prefix, expert, projection = match.groups()
        expert = int(expert)
        if projection == 'down_proj':
            key = prefix + '.w2_weight'
            return [(key, targets[key][expert])]
        key = prefix + '.w13_weight'
        mid = targets[key].shape[1] // 2
        offset = 0 if projection == 'gate_proj' else mid
        return [(key, targets[key][expert, offset:offset + mid])]
    if name.endswith('.mlp.gate.expert_bias'):
        name = name.replace('.mlp.gate.expert_bias', '.mlp.correction_bias')
    elif name.endswith('.gate_proj.weight') or name.endswith('.up_proj.weight'):
        projection = 'gate_proj' if name.endswith('.gate_proj.weight') else 'up_proj'
        key = name.replace('.' + projection + '.weight', '.gate_up_proj.weight')
        if key in targets:
            mid = targets[key].shape[0] // 2
            offset = 0 if projection == 'gate_proj' else mid
            return [(key, targets[key][offset:offset + mid])]
    if name == 'model.word_embeddings.weight':
        keys = [k for k in (name, 'model.full_word_embeddings.weight') if k in targets]
        return [(k, targets[k]) for k in keys]
    if name not in targets:
        raise KeyError(f'No resident destination for learner parameter {name}')
    return [(name, targets[name])]


@torch.no_grad()
def sync_fsdp_to_resident(model, engine, *, verify=False):
    if not isinstance(model, FSDP) or engine.parallel_mode != 'dp':
        raise ValueError('Streamed resident sync requires FSDP1 learner and DP inference')
    if engine.model.quant_config is not None:
        raise ValueError('Streamed resident sync does not support quantized inference')
    torch.cuda.synchronize(engine.device)
    begin = time.perf_counter()
    targets = dict(engine.model.named_parameters())
    targets.update(dict(engine.model.named_buffers()))
    before = {name: value.data_ptr() for name, value in targets.items()}
    covered = defaultdict(int)
    copied = 0
    units = [(name, unit) for name, unit in model.named_modules() if isinstance(unit, FSDP)]
    for path, unit in units:
        flags = []
        sources, destinations = [], []
        with FSDP.summon_full_params(unit, recurse=False, writeback=False,
                                    rank0_only=False, offload_to_cpu=False):
            prefix = _clean(path)
            prefix = prefix + '.' if prefix else ''
            for local_name, value in _managed_tensors(unit.module):
                name = prefix + _clean(local_name)
                if value.device.type != 'cuda':
                    raise RuntimeError(f'Streamed sync found CPU parameter {name}')
                for dest_name, destination in _destinations(name, value, targets):
                    if value.shape != destination.shape:
                        raise ValueError(f'Sync shape mismatch {name}: {value.shape} -> {destination.shape}')
                    sources.append(value.detach())
                    destinations.append(destination)
                    covered[dest_name] += value.numel()
                    copied += value.numel() * destination.element_size()
            # One multi-tensor copy per FSDP unit, instead of ~15k tiny launches.
            copy_groups = defaultdict(lambda: ([], []))
            for dst, src in zip(destinations, sources):
                target_group, source_group = copy_groups[(dst.dtype, src.dtype)]
                target_group.append(dst)
                source_group.append(src)
            for target_group, source_group in copy_groups.values():
                torch._foreach_copy_(target_group, source_group)
            if verify:
                flags = [torch.eq(dst, src.to(dst.dtype)).all()
                         for dst, src in zip(destinations, sources)]
            if flags and not bool(torch.stack(flags).all()):
                raise RuntimeError(f'Resident sync equality check failed in {path}')
    # h2e is an alias/cache prepared by dInfer after its source lm_head is copied.
    missing = [name for name, value in engine.model.named_parameters()
               if not name.startswith('h2e.') and covered[name] != value.numel()]
    if missing:
        raise RuntimeError(f'Incomplete resident weight sync: {missing}')
    engine.model.init_h2e_module()
    if hasattr(engine.model, 'after_processing'):
        engine.model.after_processing()
    engine._clear_routed_expert_aliases(reason='post_streamed_sync')
    for name, value in engine.model.named_parameters():
        if name in before and not name.startswith('h2e.') and before[name] != value.data_ptr():
            raise RuntimeError(f'Resident sync changed graph parameter storage: {name}')
    engine._last_sync_used_rebind = False
    engine.weights_loaded = True
    torch.cuda.synchronize(engine.device)
    return {'seconds': time.perf_counter() - begin, 'copied_bytes': copied,
            'fsdp_units': len(units), 'verified': bool(verify)}

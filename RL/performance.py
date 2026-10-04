"""Opt-in bounded CUDA traces and machine-readable wall-time phase measurements."""
import functools
import json
import os
from pathlib import Path
import time

import torch


class MicrobatchProfiler:
    def __init__(self, stage, rank):
        self.profiler = None
        directory = os.environ.get('RL_PROFILE_TRAIN_DIR')
        if directory and int(rank) == 0:
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            self.profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=int(os.environ.get("RL_PROFILE_SKIP", "20")), warmup=1, active=2, repeat=1),
                record_shapes=False,
                on_trace_ready=lambda p: p.export_chrome_trace(str(path / f'{stage}_{time.time_ns()}.json')),
            )
            self.profiler.start()

    def step(self):
        if self.profiler is not None:
            self.profiler.step()

    def close(self):
        if self.profiler is not None:
            self.profiler.stop()


def install_wall_timers(namespace, directory, rank):
    path = Path(directory) / 'performance'
    path.mkdir(parents=True, exist_ok=True)
    output = path / f'rank_{rank}.jsonl'
    run_started_at = time.time()
    names = ('run_rollouts_distributed', '_gather_episode_shards_via_files',
             '_save_training_traces', 'build_all_rows', '_build_bucketed_action_rows',
             'train_action_rows', '_safe_cleanup', '_run_pre_step_eval',
             '_get_train_state_dict_for_runtime_sync')
    for name in names:
        original = namespace[name]
        @functools.wraps(original)
        def timed(*args, _original=original, _name=name, **kwargs):
            begin = time.perf_counter()
            started = time.time()
            completed = False
            retries_before = torch.cuda.memory_stats().get('num_alloc_retries', 0) if torch.cuda.is_available() else 0
            try:
                result = _original(*args, **kwargs)
                completed = True
                return result
            finally:
                record = dict(stage=_name, rank=rank, timestamp=started, run_started_at=run_started_at,
                              completed=completed,
                              elapsed_s=time.perf_counter() - begin)
                if torch.cuda.is_available():
                    stats = torch.cuda.memory_stats()
                    record['cuda_allocation_retries'] = stats.get('num_alloc_retries', 0) - retries_before
                    record['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
                for key in ('step', 'gather_tag'):
                    if key in kwargs:
                        record[key] = kwargs[key]
                with output.open('a') as handle:
                    handle.write(json.dumps(record) + '\n')
        namespace[name] = timed
    return run_started_at


def write_run_manifest(directory, config, run_started_at):
    """Record effective runtime settings and preserve retry history."""
    from omegaconf import OmegaConf
    record = dict(run_started_at=run_started_at, config=OmegaConf.to_container(config, resolve=True),
                  torch_version=torch.__version__, cuda_version=torch.version.cuda)
    path = Path(directory) / 'performance' / f'manifest_{int(run_started_at * 1e6)}.json'
    path.write_text(json.dumps(record, indent=2) + '\n')

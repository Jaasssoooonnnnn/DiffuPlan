"""Load project-local SM100 BF16 GEMM kernels."""
import importlib.util
import os
import shutil
from pathlib import Path
import sys
import types

_BACKEND = None


def get_backend():
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    root = Path(__file__).resolve().parents[1] / '.kernel_runtime'
    package = root / 'python/deep_gemm'
    libraries = list(package.glob('_C*.so'))
    if len(libraries) != 1:
        raise RuntimeError(f'Expected one project-local DeepGEMM extension in {package}; '
                           'run bash RL/kernels/install_sm100.sh on an allocated B200 node')
    os.environ.setdefault('DG_JIT_NVCC_COMPILER', str(root / 'cuda129/bin/nvcc'))
    os.environ.setdefault('DG_JIT_CACHE_DIR', str(Path(os.environ.get('SLURM_TMPDIR', '/tmp')) / 'diffuplan_sm100'))
    compiler = os.environ.get('DIFFUPLAN_CXX') or os.environ.get('CXX') or shutil.which('c++')
    if not compiler:
        raise RuntimeError('Set DIFFUPLAN_CXX to a CUDA-compatible C++ compiler.')
    os.environ.setdefault('NVCC_PREPEND_FLAGS', '-ccbin ' + compiler)
    parent = '_diffuplan_sm100'
    if parent not in sys.modules:
        module = types.ModuleType(parent)
        module.__path__ = [str(package)]
        sys.modules[parent] = module
    name = parent + '._C'
    spec = importlib.util.spec_from_file_location(name, libraries[0])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.init(str(package))
    _BACKEND = module
    return module

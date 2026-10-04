#!/usr/bin/env bash
# Optional native BF16 expert backend; never modifies the shared dInfer environment.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RUNTIME="$ROOT/.kernel_runtime"
SRC="$RUNTIME/src/DeepGEMM"
REV=78b69000794d0937b47ae3387eff7663410264d1
PYTHON=${DIFFUPLAN_PYTHON:-python}
CONDA=${CONDA_EXE:-$(command -v conda)}
export CXX=${CXX:-$(command -v c++)}
export CC=${CC:-$(command -v cc)}
if [[ -n "${DIFFUPLAN_CPP_ENV:-}" ]]; then source "$DIFFUPLAN_CPP_ENV"; fi
if [[ ! -x "$RUNTIME/cuda129/bin/nvcc" ]]; then
    "$CONDA" create -y -p "$RUNTIME/cuda129" --override-channels \
        -c nvidia/label/cuda-12.9.1 -c nvidia -c conda-forge \
        'cuda-version=12.9' 'cuda-nvcc=12.9' 'cuda-cudart-dev=12.9' \
        'cuda-driver-dev=12.9' 'cuda-cccl=12.9' elfutils \
        libcublas-dev libcusparse-dev libcusolver-dev libcurand-dev libcufft-dev
fi
export CUDA_HOME="$RUNTIME/cuda129/targets/x86_64-linux"
mkdir -p "$CUDA_HOME/bin" "$RUNTIME/src"
[[ -e "$CUDA_HOME/lib64" ]] || ln -s lib "$CUDA_HOME/lib64"
[[ -e "$CUDA_HOME/bin/nvcc" ]] || ln -s "$RUNTIME/cuda129/bin/nvcc" "$CUDA_HOME/bin/nvcc"
if [[ ! -d "$SRC/.git" ]]; then
    git clone --recursive https://github.com/deepseek-ai/DeepGEMM.git "$SRC"
    git -C "$SRC" checkout "$REV"
    git -C "$SRC" submodule update --init --recursive
fi
[[ $(git -C "$SRC" rev-parse HEAD) == "$REV" ]] || { echo 'Unexpected DeepGEMM revision'; exit 1; }
"$PYTHON" - "$SRC" "$RUNTIME" <<'PY'
from pathlib import Path
import sys
src, runtime = map(Path, sys.argv[1:])
p = src/'csrc/python_api.cpp'
s = p.read_text()
if '#include <cuda_fp8.h>' not in s:
    p.write_text('#include <cuda_fp8.h>\n' + s)
p = src/'setup.py'
s = p.read_text()
if "'-fvisibility=hidden'" not in s:
    s = s.replace("cxx_flags = [", "cxx_flags = ['-fvisibility=hidden', '-fno-gnu-unique', ", 1)
include = repr(str(runtime/'cuda129/include'))
if include not in s:
    s = s.replace('build_include_dirs = [', 'build_include_dirs = [\n    ' + include + ',', 1)
p.write_text(s)
PY
export MAX_JOBS=${MAX_JOBS:-4}
export TORCH_CUDA_ARCH_LIST=10.0 DG_FORCE_BUILD=1
cd "$SRC"
"$PYTHON" setup.py build_ext --force bdist_wheel
"$PYTHON" -m pip install --upgrade --no-deps --target "$RUNTIME/python" dist/*.whl
"$CONDA" list -p "$RUNTIME/cuda129" --explicit > "$RUNTIME/cuda129-explicit.txt"
git diff -- setup.py csrc/python_api.cpp > "$RUNTIME/deepgemm-build.patch"
printf '%s\n' "$REV" > "$RUNTIME/deepgemm-revision.txt"

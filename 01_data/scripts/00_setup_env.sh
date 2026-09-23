#!/usr/bin/env bash
# Create the vLLM environment for the rewriting stage.
#
# Skipjack has no vLLM env yet.  Everything EXCEPT bin/03 (the GPU worker) runs under the
# existing `envs/data` interpreter, so this is only needed before production rewriting.
#
# The env lives on /weka/projects (not the 99%-full scratch quota) and its path is
# configs/cluster.yaml: env.python_vllm.
set -euo pipefail

DATA=/weka/projects/bvandur1/zhuicon1/rewrite-3b
ENVDIR=$DATA/envs/vllm
mkdir -p "$DATA/envs"

if [[ -x "$ENVDIR/bin/python" ]]; then
  echo "env already exists: $ENVDIR"
  "$ENVDIR/bin/python" -c "import vllm, transformers, pyarrow, numpy; print('vllm', vllm.__version__)"
  exit 0
fi

module load helpers/0.1.1 python/3.11.9 2>/dev/null || true
python3 -m venv "$ENVDIR"
"$ENVDIR/bin/pip" install --upgrade pip wheel

# vLLM pulls its own torch.  The 1.5B corpus was produced on vLLM 0.22.0; pin deliberately so
# the engine's inherited defaults (max_num_batched_tokens=16384, chunked prefill, prefix
# caching) are the ones recorded in configs/vllm.yaml.
"$ENVDIR/bin/pip" install \
  "vllm==0.22.0" \
  "transformers>=4.46,<5" \
  "pyarrow>=15" "numpy>=1.26,<3" "pyyaml>=6" "fasttext-wheel" "huggingface_hub>=0.25"

"$ENVDIR/bin/python" - <<'PY'
import vllm, transformers, pyarrow, numpy
print("vllm", vllm.__version__, "| transformers", transformers.__version__,
      "| pyarrow", pyarrow.__version__, "| numpy", numpy.__version__)
PY
echo "created $ENVDIR"
echo "NOTE: run bin/preflight.py under this interpreter before submitting production jobs."

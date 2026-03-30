#!/usr/bin/env bash
#SBATCH -p h200
#SBATCH --gres=gpu:8
#SBATCH --container-image=vllm/vllm-openai:v0.18.0-cu130
#SBATCH --container-mounts=/mnt/vast/hf_cache:/mnt/vast/hf_cache
#SBATCH --no-container-entrypoint
#SBATCH --job-name=server-glm5
#SBATCH --output=/mnt/home/kimbo/inferperf-260324/glm-5/logs/server.log

set -euo pipefail


pip install -U "transformers>=5.4.0"

export HF_HOME=/mnt/vast/hf_cache
export VLLM_CACHE_ROOT=/mnt/vast/hf_cache/vllm_cache
export DG_JIT_CACHE_DIR=/mnt/vast/hf_cache/deep_gemm_cache

exec vllm serve zai-org/GLM-5-FP8 \
    --tensor-parallel-size 8 \
    --speculative-config.method mtp \
    --speculative-config.num_speculative_tokens 3 \
    --max-model-len 65536 \
    --gpu-memory-utilization 0.85 \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --enable-auto-tool-choice \
    --host 0.0.0.0 \
    --port 8000

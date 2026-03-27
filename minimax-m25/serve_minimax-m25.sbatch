#!/usr/bin/env bash
#SBATCH -p h200
#SBATCH --gres=gpu:8
#SBATCH --container-image=vllm/vllm-openai:v0.18.0-cu130
#SBATCH --container-mounts=/mnt/vast/hf_cache:/mnt/vast/hf_cache
#SBATCH --no-container-entrypoint
#SBATCH --job-name=server-mmm25
#SBATCH --output=/mnt/home/kimbo/inferperf-260324/minimax-m25/logs/server.log

set -euo pipefail

export HF_HOME=/mnt/vast/hf_cache
export VLLM_CACHE_ROOT=/mnt/vast/hf_cache/vllm_cache
export VLLM_USE_DEEP_GEMM=0

exec vllm serve MiniMaxAI/MiniMax-M2.5 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --tool-call-parser minimax_m2 \
    --reasoning-parser minimax_m2_append_think \
    --enable-auto-tool-choice \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000

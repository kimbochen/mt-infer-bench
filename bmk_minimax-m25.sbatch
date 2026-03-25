#!/usr/bin/env bash
#SBATCH -p h200
#SBATCH --gres=gpu:0
#SBATCH --container-image=vllm/vllm-openai:v0.18.0-cu130
#SBATCH --container-mounts=/mnt/home/kimbo/inferperf-260324:/workspace,/mnt/vast/hf_cache:/mnt/vast/hf_cache
#SBATCH --no-container-entrypoint
#SBATCH --job-name=mtbench-mmm25
#SBATCH --output=/mnt/home/kimbo/inferperf-260324/logs/client.log

set -euo pipefail

# Install dependencies (if needed)
pip install -q aiohttp transformers tqdm

# Check for SERVER_HOST env var
if [ -z "${SERVER_HOST:-}" ]; then
    echo "ERROR: SERVER_HOST not set. Submit with:"
    echo "  SERVER_HOST=\$(squeue --name=vllm-minimax-m25 --format=%N -h) sbatch bmk_minimax-m25.sbatch"
    exit 1
fi
SERVER_URL="http://${SERVER_HOST}:8000"

# Wait for server to be healthy
echo "Waiting for server at $SERVER_URL..."
while ! curl -sf "${SERVER_URL}/health" > /dev/null; do
    sleep 30
done
echo "Server is ready"


# Set variables
export HF_HOME=/mnt/vast/hf_cache

MODEL="MiniMaxAI/MiniMax-M2.5"
BENCH_DIR=/workspace/multiturn_benchmark
RESULTS_DIR=/workspace/results
SPEC_FILE="${BENCH_DIR}/wildchat_downsample_40k.json"

mkdir -p "$RESULTS_DIR"

REQUEST_RATE=0.04
CONVOS_PER_CLIENT=4
BATCHES_PER_CLIENT=10  # each client processes 10x its active window


# Sweep number of clients
NUM_CLIENTS_SWEEP=(1 2 4 8 16 32 64 128 192 256 384 512 768 1024)

for NC in "${NUM_CLIENTS_SWEEP[@]}"; do
    JSON_FILE="${RESULTS_DIR}/mtbench_minimax-m25_clients${NC}.json"
    if [ -f "$JSON_FILE" ]; then
        echo "Skipping (exists): $JSON_FILE"
        continue
    fi

    MAX_ACTIVE=$((NC * CONVOS_PER_CLIENT))
    MAX_CONVOS=$((NC * CONVOS_PER_CLIENT * BATCHES_PER_CLIENT))
    AGG_RATE=$(python3 -c "print($NC * $REQUEST_RATE)")
    echo "=== Benchmark: num-clients=${NC}, max-active=${MAX_ACTIVE}, max-convos=${MAX_CONVOS}, agg-request-rate=${AGG_RATE} req/s ==="

    python3 ${BENCH_DIR}/benchmark_serving_multi_turn.py \
        --model $MODEL \
        --url "$SERVER_URL" \
        --input-file "$SPEC_FILE" \
        --num-clients $NC \
        --max-active-conversations $MAX_ACTIVE \
        --max-conversations $MAX_CONVOS \
        --request-rate $REQUEST_RATE \
        --no-early-stop \
        --json-output "$JSON_FILE"

    echo "Done: $JSON_FILE"
done

echo "All benchmarks complete!"

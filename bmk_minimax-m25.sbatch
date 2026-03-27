#!/usr/bin/env bash
#SBATCH -p h200
#SBATCH --gres=gpu:0
#SBATCH --container-image=vllm/vllm-openai:v0.18.0-cu130
#SBATCH --container-mounts=/mnt/home/kimbo/inferperf-260324:/workspace,/mnt/vast/hf_cache:/mnt/vast/hf_cache
#SBATCH --no-container-entrypoint
#SBATCH --job-name=mtbench-mmm25
#SBATCH --output=/mnt/home/kimbo/inferperf-260324/logs/client.log

set -euo pipefail

: "${SERVER_HOST:?SERVER_HOST not set. Use: SERVER_HOST=<node> sbatch bmk_minimax-m25.sbatch}"
SERVER_URL="http://${SERVER_HOST}:8000"


# Wait for server to be healthy
echo "Waiting for server at $SERVER_URL..."
while ! curl -sf "${SERVER_URL}/health" > /dev/null; do
    sleep 30
done
echo "Server is ready"


# Set variables
MODEL="MiniMaxAI/MiniMax-M2.5"
HF_HOME=/mnt/vast/hf_cache
BENCH_DIR=/workspace/multiturn_benchmark
RESULTS_DIR=/workspace/results
SPEC_FILE="${BENCH_DIR}/wildchat_downsample_15k.json"

export HF_HOME
mkdir -p "$RESULTS_DIR"


# Sweep number of clients
NUM_CLIENTS_SWEEP=(16 32 48 64 96 128 192 256 384 512)
REQUEST_RATE=0.04

for NC in "${NUM_CLIENTS_SWEEP[@]}"; do
    JSON_FILE="${RESULTS_DIR}/mtbench_minimax-m25_clients${NC}.json"
    if [ -f "$JSON_FILE" ]; then
        echo "Skipping (exists): $JSON_FILE"
        continue
    fi

    MAX_ACTIVE=$NC # 1 session per client, no interleaving

    # Total sessions = NC clients * 0.04 req/s * 10 min steady state * 60 s/min
    # Minimum 1000 sessions for P99 percentile stability
    MAX_SESSIONS=$(python3 -c "import math; print(max(1000, math.ceil($NC * $REQUEST_RATE * 10 * 60)))")

    echo "=== Benchmark: num-clients=${NC}, max-active=${MAX_ACTIVE}, max-sessions=${MAX_SESSIONS} ==="

    python3 ${BENCH_DIR}/bench_serving_multiturn.py \
        --model $MODEL \
        --url "$SERVER_URL" \
        --input-file "$SPEC_FILE" \
        --num-clients $NC \
        --max-active-sessions $MAX_ACTIVE \
        --max-sessions $MAX_SESSIONS \
        --request-rate $REQUEST_RATE \
        --json-output "$JSON_FILE"

    echo "Done: $JSON_FILE"
done

echo "All benchmarks complete!"

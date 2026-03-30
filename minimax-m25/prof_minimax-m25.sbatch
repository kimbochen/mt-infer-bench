#!/usr/bin/env bash
#SBATCH -p h200
#SBATCH --gres=gpu:8
#SBATCH --container-image=vllm/vllm-openai:v0.18.0-cu130
#SBATCH --container-mounts=/mnt/home/kimbo/inferperf-260324:/workspace,/mnt/vast/hf_cache:/mnt/vast/hf_cache
#SBATCH --no-container-entrypoint
#SBATCH --job-name=prof-torch-mmm25
#SBATCH --output=/mnt/home/kimbo/inferperf-260324/minimax-m25/logs/prof_torch.log

set -euo pipefail

SERVER_URL="http://localhost:8000"
LOG_DIR=/workspace/minimax-m25/logs
TRACE_BASE=/workspace/minimax-m25/traces
BENCH_DIR=/workspace/multiturn_benchmark

mkdir -p "$LOG_DIR"

export HF_HOME=/mnt/vast/hf_cache
export VLLM_CACHE_ROOT=/mnt/vast/hf_cache/vllm_cache
export VLLM_USE_DEEP_GEMM=0
export VLLM_CUSTOM_SCOPES_FOR_PROFILING=1

MODEL="MiniMaxAI/MiniMax-M2.5"
SPEC_FILE="${BENCH_DIR}/wildchat_downsample_15k.json"
REQUEST_RATE=0.04

PROFILE_NCS=(64 256)
WARMUP_SECS=120
POLL_INTERVAL=5
POLL_TIMEOUT=120

for NC in "${PROFILE_NCS[@]}"; do
    TRACE_DIR="${TRACE_BASE}/traces_minimax-m25_nc${NC}"
    mkdir -p "$TRACE_DIR"

    echo "=== Profiling NC=${NC} (torch profiler) ==="

    # Start server with torch profiler config
    echo "Starting server (traces -> ${TRACE_DIR})..."
    vllm serve $MODEL \
        --tensor-parallel-size 8 \
        --enable-expert-parallel \
        --tool-call-parser minimax_m2 \
        --reasoning-parser minimax_m2_append_think \
        --enable-auto-tool-choice \
        --trust-remote-code \
        --host 0.0.0.0 \
        --port 8000 \
        --profiler-config "{
            \"profiler\": \"torch\",
            \"torch_profiler_dir\": \"${TRACE_DIR}\",
            \"delay_iterations\": 0,
            \"wait_iterations\": 0,
            \"warmup_iterations\": 3,
            \"active_iterations\": 10
        }" \
        > "${LOG_DIR}/prof_torch_server_nc${NC}.log" 2>&1 &
    SERVER_PID=$!

    # Wait for server to be healthy
    echo "Waiting for server..."
    while ! curl -sf "${SERVER_URL}/health" > /dev/null 2>&1; do
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "Server died. Check ${LOG_DIR}/prof_torch_server_nc${NC}.log"
            exit 1
        fi
        sleep 10
    done
    echo "Server is ready"

    # Launch benchmark in background
    MAX_ACTIVE=$NC
    MAX_SESSIONS=$(python3 -c "import math; print(max(1000, math.ceil($NC * $REQUEST_RATE * 10 * 60)))")

    echo "Starting benchmark (NC=${NC}, sessions=${MAX_SESSIONS})..."
    python3 ${BENCH_DIR}/bench_serving_multiturn.py \
        --model $MODEL \
        --url "$SERVER_URL" \
        --input-file "$SPEC_FILE" \
        --num-clients $NC \
        --max-active-sessions $MAX_ACTIVE \
        --max-sessions $MAX_SESSIONS \
        --request-rate $REQUEST_RATE \
        --json-output "/tmp/prof_discard_nc${NC}.json" \
        > "${LOG_DIR}/prof_torch_client_nc${NC}.log" 2>&1 &
    BMK_PID=$!

    # Wait for steady state
    echo "Waiting ${WARMUP_SECS}s for steady state..."
    sleep $WARMUP_SECS

    # Trigger torch profiler
    echo "Triggering profiler (3 warmup + 10 active steps)..."
    curl -sf -X POST "${SERVER_URL}/start_profile" && echo " -> Profiler started" || echo " -> Failed to start profiler"

    # Wait for profiler to complete
    echo "Waiting for profiler to complete..."
    ELAPSED=0
    while [ $ELAPSED -lt $POLL_TIMEOUT ]; do
        sleep $POLL_INTERVAL
        ELAPSED=$((ELAPSED + POLL_INTERVAL))
        echo "  ${ELAPSED}s elapsed..."
    done
    curl -sf -X POST "${SERVER_URL}/stop_profile" && echo " -> Profiler stopped/flushed" || true

    # Kill benchmark and server
    echo "Stopping benchmark and server..."
    kill $BMK_PID 2>/dev/null || true
    wait $BMK_PID 2>/dev/null || true
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true

    echo "Done: NC=${NC} -> ${TRACE_DIR}"
    echo ""
done

echo "All profiling runs complete!"
echo "Traces:"
for NC in "${PROFILE_NCS[@]}"; do
    echo "  NC=${NC}: minimax-m25/traces/traces_minimax-m25_nc${NC}/"
done

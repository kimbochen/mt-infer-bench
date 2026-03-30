#!/usr/bin/env bash
#SBATCH -p h200
#SBATCH --gres=gpu:8
#SBATCH --container-image=vllm/vllm-openai:v0.18.0-cu130
#SBATCH --container-mounts=/mnt/home/kimbo/inferperf-260324:/workspace,/mnt/vast/hf_cache:/mnt/vast/hf_cache
#SBATCH --no-container-entrypoint
#SBATCH --job-name=prof-glm5
#SBATCH --output=/mnt/home/kimbo/inferperf-260324/glm-5/logs/prof.log

set -euo pipefail

SERVER_URL="http://localhost:8000"
LOG_DIR=/workspace/glm-5/logs
TRACE_BASE=/workspace/glm-5/traces
BENCH_DIR=/workspace/multiturn_benchmark

mkdir -p "$LOG_DIR"

pip install --upgrade transformers

export HF_HOME=/mnt/vast/hf_cache
export VLLM_CACHE_ROOT=/mnt/vast/hf_cache/vllm_cache
export DG_JIT_CACHE_DIR=/mnt/vast/hf_cache/deep_gemm_cache
export VLLM_CUSTOM_SCOPES_FOR_PROFILING=1

MODEL="zai-org/GLM-5-FP8"
SPEC_FILE="${BENCH_DIR}/wildchat_downsample_15k.json"
REQUEST_RATE=0.04

PROFILE_NCS=(64 256)
WARMUP_SECS=120
POLL_INTERVAL=5
POLL_TIMEOUT=120

for NC in "${PROFILE_NCS[@]}"; do
    TRACE_DIR="${TRACE_BASE}/traces_glm-5_nc${NC}"
    mkdir -p "$TRACE_DIR"

    echo "=== Profiling NC=${NC} (torch profiler) ==="

    # Start server with torch profiler config
    echo "Starting server (traces -> ${TRACE_DIR})..."
    vllm serve $MODEL \
        --tensor-parallel-size 8 \
        --speculative-config.method mtp \
        --speculative-config.num_speculative_tokens 1 \
        --tool-call-parser glm47 \
        --reasoning-parser glm45 \
        --enable-auto-tool-choice \
        --max-model-len 65536 \
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
        > "${LOG_DIR}/prof_server_nc${NC}.log" 2>&1 &
    SERVER_PID=$!

    # Wait for server to be healthy
    echo "Waiting for server..."
    while ! curl -sf "${SERVER_URL}/health" > /dev/null 2>&1; do
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "Server died. Check ${LOG_DIR}/prof_server_nc${NC}.log"
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
        > "${LOG_DIR}/prof_client_nc${NC}.log" 2>&1 &
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
    echo "  NC=${NC}: glm-5/traces/traces_glm-5_nc${NC}/"
done

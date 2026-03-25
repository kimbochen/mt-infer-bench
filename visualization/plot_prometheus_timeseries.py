"""Plot Prometheus metrics time series for a single benchmark data point.

4-panel dashboard:
1. Load: num_requests_running + num_requests_waiting
2. Throughput: prompt tokens/s + generation tokens/s
3. Cache: prefix cache hit rate over time
4. Latency breakdown: rolling avg queue_time, prefill_time, decode_time per request
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np


def load_data(filepath: str) -> dict:
    with open(filepath) as f:
        return json.load(f)


def find_key(metrics: dict, substr: str, exclude: list[str] | None = None) -> str | None:
    """Find a metric key containing substr, excluding keys with any exclude strings."""
    if exclude is None:
        exclude = ["_created", "_bucket"]
    for k in metrics:
        if substr in k and all(e not in k for e in exclude):
            return k
    return None


def extract_gauge(snapshots: list[dict], substr: str) -> tuple[np.ndarray, np.ndarray]:
    """Extract timestamps and values for a gauge metric."""
    key = find_key(snapshots[0]["metrics"], substr)
    if not key:
        return np.array([]), np.array([])
    ts = np.array([s["timestamp"] for s in snapshots])
    vals = np.array([s["metrics"].get(key, 0) for s in snapshots])
    return ts, vals


def extract_counter_rate(
    snapshots: list[dict], substr: str, window: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """Extract timestamps and per-second rate for a counter metric."""
    key = find_key(snapshots[0]["metrics"], substr)
    if not key:
        return np.array([]), np.array([])
    ts = np.array([s["timestamp"] for s in snapshots])
    vals = np.array([s["metrics"].get(key, 0) for s in snapshots])

    # Compute rate as delta/dt, smoothed over window
    dt = np.diff(ts)
    dv = np.diff(vals)
    rate = np.where(dt > 0, dv / dt, 0)

    # Smooth with rolling average
    if len(rate) > window:
        kernel = np.ones(window) / window
        rate = np.convolve(rate, kernel, mode="same")

    return ts[1:], rate


def extract_counter_ratio(
    snapshots: list[dict], num_substr: str, den_substr: str, window: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Extract rolling ratio of two counter deltas."""
    num_key = find_key(snapshots[0]["metrics"], num_substr)
    den_key = find_key(snapshots[0]["metrics"], den_substr)
    if not num_key or not den_key:
        return np.array([]), np.array([])
    ts = np.array([s["timestamp"] for s in snapshots])
    nums = np.array([s["metrics"].get(num_key, 0) for s in snapshots])
    dens = np.array([s["metrics"].get(den_key, 0) for s in snapshots])

    d_num = np.diff(nums)
    d_den = np.diff(dens)

    # Rolling sum for smoothing
    if len(d_num) > window:
        kernel = np.ones(window)
        d_num_smooth = np.convolve(d_num, kernel, mode="same")
        d_den_smooth = np.convolve(d_den, kernel, mode="same")
    else:
        d_num_smooth = d_num
        d_den_smooth = d_den

    ratio = np.where(d_den_smooth > 0, d_num_smooth / d_den_smooth * 100, 0)
    return ts[1:], ratio


def extract_per_request_latency(
    snapshots: list[dict], sum_substr: str, count_substr: str, window: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Extract rolling per-request latency (ms) from sum/count counters."""
    sum_key = find_key(snapshots[0]["metrics"], sum_substr)
    count_key = find_key(snapshots[0]["metrics"], count_substr)
    if not sum_key or not count_key:
        return np.array([]), np.array([])
    ts = np.array([s["timestamp"] for s in snapshots])
    sums = np.array([s["metrics"].get(sum_key, 0) for s in snapshots])
    counts = np.array([s["metrics"].get(count_key, 0) for s in snapshots])

    d_sum = np.diff(sums)
    d_count = np.diff(counts)

    if len(d_sum) > window:
        kernel = np.ones(window)
        d_sum = np.convolve(d_sum, kernel, mode="same")
        d_count = np.convolve(d_count, kernel, mode="same")

    # Convert seconds to milliseconds
    latency_ms = np.where(d_count > 0, d_sum / d_count * 1000, 0)
    return ts[1:], latency_ms


def main():
    parser = argparse.ArgumentParser(
        description="Plot Prometheus time series for a single benchmark run"
    )
    parser.add_argument(
        "json_file",
        type=str,
        help="Benchmark result JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image file (default: prometheus_nc{NC:04d}.png)",
    )
    args = parser.parse_args()

    data = load_data(args.json_file)
    prom = data.get("prometheus", [])
    if len(prom) < 3:
        print("Not enough prometheus snapshots")
        return

    nc = data["params"]["num_clients"]
    output_path = args.output or f"prometheus_nc{nc:04d}.png"

    # Normalize timestamps to start at 0 (minutes)
    t0 = prom[0]["timestamp"]
    for s in prom:
        s["timestamp"] = (s["timestamp"] - t0) / 60.0  # minutes

    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

    # Panel 1: Load
    ax = axes[0]
    ts_r, running = extract_gauge(prom, "num_requests_running")
    ts_w, waiting = extract_gauge(prom, "num_requests_waiting")
    if len(ts_r):
        ax.plot(ts_r, running, color="#2166ac", linewidth=1, label="Running")
    if len(ts_w):
        ax.plot(ts_w, waiting, color="#b2182b", linewidth=1, label="Waiting")
    ax.set_ylabel("Requests")
    ax.set_title("Server Load")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Panel 2: Throughput
    ax = axes[1]
    ts_p, prompt_rate = extract_counter_rate(prom, "prompt_tokens_total")
    ts_g, gen_rate = extract_counter_rate(prom, "generation_tokens_total")
    if len(ts_p):
        ax.plot(ts_p, prompt_rate, color="#2166ac", linewidth=1, label="Prefill tok/s")
    if len(ts_g):
        ax.plot(ts_g, gen_rate, color="#b2182b", linewidth=1, label="Decode tok/s")
    ax.set_ylabel("Tokens/s")
    ax.set_title("Token Throughput")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Panel 3: Cache hit rate
    ax = axes[2]
    ts_c, cache_rate = extract_counter_ratio(
        prom, "prefix_cache_hits_total", "prefix_cache_queries_total"
    )
    if len(ts_c):
        ax.plot(ts_c, cache_rate, color="#2166ac", linewidth=1)
    ax.set_ylabel("Cache Hit Rate (%)")
    ax.set_title("Prefix Cache Hit Rate")
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)

    # Panel 4: Latency breakdown
    ax = axes[3]
    ts_q, queue_ms = extract_per_request_latency(
        prom, "request_queue_time_seconds_sum", "request_queue_time_seconds_count"
    )
    ts_pf, prefill_ms = extract_per_request_latency(
        prom, "request_prefill_time_seconds_sum", "request_prefill_time_seconds_count"
    )
    ts_d, decode_ms = extract_per_request_latency(
        prom, "request_decode_time_seconds_sum", "request_decode_time_seconds_count"
    )
    if len(ts_q):
        ax.plot(ts_q, queue_ms, color="#f4a582", linewidth=1, label="Queue")
    if len(ts_pf):
        ax.plot(ts_pf, prefill_ms, color="#2166ac", linewidth=1, label="Prefill")
    if len(ts_d):
        ax.plot(ts_d, decode_ms, color="#b2182b", linewidth=1, label="Decode")
    ax.set_ylabel("Latency (ms/req)")
    ax.set_xlabel("Time (min)")
    ax.set_title("Per-Request Latency Breakdown")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"MiniMax M2.5 — Prometheus Metrics (NC={nc})",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()

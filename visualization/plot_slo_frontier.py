"""Plot SLO-to-throughput charts from benchmark sweep results.

Chart 1: TTFT SLO (ms) vs Input Throughput (tok/gpu/s)
Chart 2: TPOT SLO (ms) vs Output Throughput (tok/gpu/s)

Each chart has three lines: SLO P50, SLO P90, SLO P99.
"""

import argparse
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_sweep(results_dir: str) -> list[dict]:
    """Load all sweep JSON files, sorted by num_clients."""
    files = glob.glob(str(Path(results_dir) / "mtbench_*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON results found in {results_dir}")

    runs = []
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        raw = d["raw_requests"]
        runs.append({
            "num_clients": d["params"]["num_clients"],
            "raw": raw,
            "prometheus": d.get("prometheus", []),
        })

    runs.sort(key=lambda r: r["num_clients"])
    return runs


def _get_prom_delta(prom: list[dict], substr: str) -> float:
    """Get delta of a prometheus counter between first and last snapshot."""
    if len(prom) < 2:
        return 0
    first, last = prom[0]["metrics"], prom[-1]["metrics"]
    keys = [k for k in last if substr in k and "_created" not in k]
    if keys:
        return last[keys[0]] - first.get(keys[0], 0)
    return 0


def compute_sweep_points(
    runs: list[dict],
    metric_key: str,
    token_key: str,
    num_gpus: int,
    percentile: float,
) -> list[dict]:
    """Return sweep point data including percentile, throughput, and token breakdown."""
    points = []
    for run in runs:
        raw = run["raw"]
        if len(raw) == 0:
            continue

        metric_vals = np.array([r[metric_key] for r in raw])
        token_vals = np.array([r[token_key] for r in raw])
        start_times = np.array([r["start_time_ms"] for r in raw])
        latencies = np.array([r["latency_ms"] for r in raw])

        pct_val = np.percentile(metric_vals, percentile)
        runtime_sec = (
            (start_times + latencies).max() - start_times.min()
        ) / 1000.0

        if runtime_sec <= 0:
            continue

        tput = token_vals.sum() / runtime_sec / num_gpus

        # Per-request token breakdown from prometheus
        prom = run.get("prometheus", [])
        n = len(raw)
        dp = _get_prom_delta(prom, "prompt_tokens_total")
        dc = _get_prom_delta(prom, "prompt_tokens_cached_total")
        if dp > 0 and n > 0:
            base_per_req = int((dp - dc) / n)
            cached_per_req = int(dc / n)
        else:
            all_input = np.array([r["input_num_tokens"] for r in raw])
            all_cached_frac = np.array([r["approx_cached_percent"] for r in raw])
            cached_per_req = int((all_input * all_cached_frac / 100.0).mean())
            base_per_req = int(all_input.mean() - cached_per_req)

        dg = _get_prom_delta(prom, "generation_tokens_total")
        output_per_req = int(dg / n) if dg > 0 and n > 0 else int(
            np.mean([r["output_num_tokens"] for r in raw])
        )

        points.append({
            "x": pct_val,
            "y": tput,
            "nc": run["num_clients"],
            "base_per_req": base_per_req,
            "cached_per_req": cached_per_req,
            "output_per_req": output_per_req,
        })

    return points


def pareto_frontier(points: list[dict]) -> list[dict]:
    """Extract Pareto-optimal points: max throughput (Y) for increasing latency (X)."""
    if not points:
        return []
    sorted_pts = sorted(points, key=lambda p: p["x"])
    frontier = []
    max_y = -float("inf")
    for pt in sorted_pts:
        if pt["y"] > max_y:
            frontier.append(pt)
            max_y = pt["y"]
    return frontier


def compute_spec_token_stats(spec_file: str) -> tuple[float, float, float]:
    """Compute average per-request token stats from the offline spec."""
    with open(spec_file) as f:
        spec = json.load(f)

    total_new = 0
    total_cached = 0
    total_output = 0
    total_requests = 0

    sessions = spec.get("sessions", spec.get("conversations", spec))
    for conv in sessions:
        inp = conv["input_tokens"]
        out = conv["output_tokens"]
        n = conv["num_turns"]
        for k in range(n):
            total_new += inp[k]
            total_cached += sum(inp[:k]) + sum(out[:k])
            total_output += out[k]
            total_requests += 1

    return (
        total_new / total_requests,
        total_cached / total_requests,
        total_output / total_requests,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Plot SLO-to-throughput charts from benchmark sweep results"
    )
    parser.add_argument(
        "results_dir",
        type=str,
        help="Directory containing mtbench_*.json result files",
    )
    parser.add_argument(
        "--spec-file",
        type=str,
        required=True,
        help="Offline spec JSON file (for computing token stats in the title)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model name for the title (e.g., 'MiniMax M2.5')",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        required=True,
        help="Number of GPUs used for serving",
    )
    parser.add_argument(
        "--parallelism",
        type=str,
        default="TP",
        help="Parallelism strategy label (e.g., 'TP', 'TEP', 'PP')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image file",
    )
    args = parser.parse_args()

    output_path = args.output or f"slo_frontier.png"

    runs = load_sweep(args.results_dir)
    print(f"Loaded {len(runs)} sweep points: "
          f"NC={[r['num_clients'] for r in runs]}")

    avg_new_input, avg_cached, avg_output = compute_spec_token_stats(args.spec_file)

    title = f"SLO Frontier - {args.model_name} {args.num_gpus}xH200 {args.parallelism}{args.num_gpus}"
    subtitle = (
        f"WildChat Multi-Turn, Avg. per Turn: Input {avg_new_input:.0f} tok | "
        f"Cached Input {avg_cached:.0f} tok | Output {avg_output:.0f} tok"
    )

    # Blue-to-red progression: easier SLO (P50) is cool, harder (P99) is hot
    percentiles = [
        ("P50", 50, "#2166ac"),
        ("P90", 90, "#f4a582"),
        ("P99", 99, "#b2182b"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    def plot_chart(ax, metric_key, token_key, xlabel, ylabel,
                   label_fn=None, label_desc=None):
        for label, pct, color in percentiles:
            pts = compute_sweep_points(
                runs, metric_key, token_key, args.num_gpus, pct
            )
            if pts:
                xs = [p["x"] for p in pts]
                ys = [p["y"] for p in pts]
                ax.scatter(xs, ys, color=color, s=30, zorder=2)
                front = pareto_frontier(pts)
                if len(front) >= 2:
                    fx = [p["x"] for p in front]
                    fy = [p["y"] for p in front]
                    ax.plot(fx, fy, color=color, linewidth=1.5, alpha=0.7,
                            label=f"SLO {label}", zorder=1)
                else:
                    ax.scatter([], [], color=color, s=30, label=f"SLO {label}")

                if pct == 99 and label_fn is not None:
                    frontier_pts = set(
                        (p["x"], p["y"]) for p in pareto_frontier(pts)
                    )
                    for pt in pts:
                        if (pt["x"], pt["y"]) in frontier_pts:
                            ax.annotate(
                                label_fn(pt),
                                (pt["x"], pt["y"]),
                                textcoords="offset points",
                                xytext=(-5, 6),
                                fontsize=7,
                                color="#555555",
                                ha="right",
                            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
        if label_desc:
            ax.text(
                0.0, 1.02, label_desc,
                transform=ax.transAxes, fontsize=7, color="#888888",
                verticalalignment="bottom", horizontalalignment="left",
            )

    plot_chart(
        ax1, "ttft_ms", "input_num_tokens",
        "TTFT (ms)", "Input Throughput (tok/gpu/s)",
        label_fn=lambda pt: f'{pt["base_per_req"]} tok / {pt["cached_per_req"]} tok',
        label_desc="Label: Avg. Input per Turn / Avg. Cached Input per Turn",
    )
    plot_chart(
        ax2, "tpot_ms", "output_num_tokens",
        "TPOT (ms)", "Output Throughput (tok/gpu/s)",
    )

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.05)
    fig.text(0.5, 0.99, subtitle, ha="center", fontsize=11, color="#555555")
    fig.tight_layout()
    fig.subplots_adjust(top=0.90)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()

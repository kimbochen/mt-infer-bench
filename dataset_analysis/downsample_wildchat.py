"""Downsample wildchat_stats.json while preserving the distribution of
num_turns, input tokens, and output tokens as closely as possible.

Strategy:
  1. Stratify sessions by (num_turns_bin, total_input_tok_bin,
     total_output_tok_bin) using quantile-based binning on all dimensions.
  2. Adaptively reduce bin granularity so the number of populated strata
     stays well below n_samples.
  3. Allocate the target sample count proportionally across strata
     (largest-remainder method to hit the exact target).
  4. Random-sample within each stratum.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


STATS_PATH = Path(__file__).resolve().parent / 'wildchat_stats.json'


def quantile_bin(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Assign each value to a quantile-based bin index in [0, n_bins)."""
    if len(values) == 0:
        return np.array([], dtype=int)
    edges = np.percentile(values, np.linspace(0, 100, n_bins + 1))
    bins = np.searchsorted(edges[1:-1], values, side='right')
    return bins


def largest_remainder_round(fractions: np.ndarray, total: int) -> np.ndarray:
    """Round fractions to integers that sum exactly to `total`."""
    floors = np.floor(fractions).astype(int)
    remainders = fractions - floors
    shortfall = total - floors.sum()
    indices = np.argsort(-remainders)
    for i in range(int(shortfall)):
        floors[indices[i]] += 1
    return floors


def downsample(stats: list[dict], n_samples: int, seed: int = 42,
               tok_bins: int = 5) -> list[dict]:
    """Return a subset of `stats` of size `n_samples` that preserves the
    joint distribution of (num_turns, input tokens, output tokens)."""
    rng = np.random.default_rng(seed)
    n = len(stats)
    if n_samples >= n:
        return stats

    # Compute per-session features for stratification
    num_turns = np.array([s['num_turns'] for s in stats])
    total_in = np.array([sum(s['input_toks']) for s in stats])
    total_out = np.array([sum(s['output_toks']) for s in stats])

    # Adaptively choose bin counts so total strata < n_samples / 2
    # Total strata <= turn_bins * tok_bins * tok_bins
    max_strata = max(n_samples // 2, 20)
    turn_bins = min(tok_bins, max(2, int(max_strata ** (1/3))))
    effective_tok_bins = min(tok_bins, max(2, int((max_strata / turn_bins) ** 0.5)))

    # Bin all dimensions with quantile-based bins
    turn_binned = quantile_bin(num_turns.astype(float), turn_bins)
    in_bins = quantile_bin(total_in, effective_tok_bins)
    out_bins = quantile_bin(total_out, effective_tok_bins)

    # Build strata
    strata: dict[tuple, list[int]] = defaultdict(list)
    for i in range(n):
        key = (int(turn_binned[i]), int(in_bins[i]), int(out_bins[i]))
        strata[key].append(i)

    # Proportional allocation
    keys = list(strata.keys())
    sizes = np.array([len(strata[k]) for k in keys], dtype=float)
    raw_alloc = sizes / sizes.sum() * n_samples
    alloc = largest_remainder_round(raw_alloc, n_samples)

    # Sample from each stratum
    sampled_indices = []
    for k, count in zip(keys, alloc):
        pool = strata[k]
        count = min(count, len(pool))
        if count > 0:
            chosen = rng.choice(pool, size=count, replace=False)
            sampled_indices.extend(chosen.tolist())

    # Shuffle final result
    rng.shuffle(sampled_indices)
    return [stats[i] for i in sampled_indices]


def main():
    parser = argparse.ArgumentParser(
        description='Downsample wildchat_stats.json preserving distributions')
    parser.add_argument('-n', '--n-samples', type=int, default=1000,
                        help='Number of samples in the output (default: 1000)')
    parser.add_argument('--min-samples', type=int, default=100,
                        help='Minimum allowed sample count (default: 100)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--tok-bins', type=int, default=5,
                        help='Number of quantile bins for token stratification (default: 5)')
    parser.add_argument('--input', type=str, default=str(STATS_PATH),
                        help='Path to input wildchat_stats.json')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to output JSON (default: wildchat_downsampled_{n}.json in cwd)')
    args = parser.parse_args()

    if args.n_samples < args.min_samples:
        parser.error(f'--n-samples ({args.n_samples}) must be >= --min-samples ({args.min_samples})')

    print(f'Loading {args.input} ...')
    with open(args.input) as f:
        stats = json.load(f)
    print(f'Loaded {len(stats)} sessions.')

    result = downsample(stats, args.n_samples, seed=args.seed,
                        tok_bins=args.tok_bins)
    print(f'Downsampled to {len(result)} sessions.')

    # Convert to offline spec format, padding short token lists with 1
    conversations = []
    padded = 0
    for s in result:
        n = s['num_turns']
        inp = list(s['input_toks'])
        out = list(s['output_toks'])
        if len(inp) < n or len(out) < n:
            padded += 1
            while len(inp) < n:
                inp.append(1)
            while len(out) < n:
                out.append(1)
        conversations.append({
            'num_turns': n,
            'input_tokens': inp[:n],
            'output_tokens': out[:n],
        })
    if padded:
        print(f'Padded {padded} sessions with short token lists (filled with 1).')
    output_data = {
        'text_files': ['/workspace/multiturn_benchmark/gutenberg_11m.txt'],
        'conversations': conversations,
    }

    out_path = args.output or f'wildchat_downsampled_{len(result)}.json'
    with open(out_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f'Saved to {out_path}')

    # Print distribution comparison
    print('\n--- Distribution comparison ---')
    orig_turns = [s['num_turns'] for s in stats]
    down_turns = [c['num_turns'] for c in conversations]
    orig_in = [t for s in stats for t in s['input_toks']]
    down_in = [t for c in conversations for t in c['input_tokens']]
    orig_out = [t for s in stats for t in s['output_toks']]
    down_out = [t for c in conversations for t in c['output_tokens']]

    for name, orig, down in [('num_turns', orig_turns, down_turns),
                              ('input_tokens', orig_in, down_in),
                              ('output_tokens', orig_out, down_out)]:
        o, d = np.array(orig, dtype=float), np.array(down, dtype=float)
        print(f'\n{name}:')
        print(f'  Original  — mean={o.mean():.2f}  median={np.median(o):.1f}  '
              f'p5={np.percentile(o,5):.0f}  p95={np.percentile(o,95):.0f}')
        print(f'  Downsampl — mean={d.mean():.2f}  median={np.median(d):.1f}  '
              f'p5={np.percentile(d,5):.0f}  p95={np.percentile(d,95):.0f}')


if __name__ == '__main__':
    main()

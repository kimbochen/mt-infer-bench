import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from scipy.stats import lognorm, geom
from tqdm import tqdm
import tiktoken


def collect_stats(stats_filepath):
    if stats_filepath.exists():
        with open(stats_filepath) as f:
            stats = json.load(f)
        return stats

    dataset = load_dataset('allenai/WildChat-4.8M', split='train', cache_dir='/mnt/vast/hf_cache')

    def process_row(row):
        enc = tiktoken.get_encoding('cl100k_base')
        conv = row['conversation']
        user_msgs = [msg['content'] for msg in conv if msg['content'] and msg['role'] == 'user']
        assistant_msgs = [msg['content'] for msg in conv if msg['content'] and msg['role'] == 'assistant']
        if not user_msgs or not assistant_msgs:
            return {'input_toks': [], 'output_toks': [], 'num_turns': 0}
        return {
            'input_toks': [len(ids) for ids in enc.encode_batch(user_msgs, disallowed_special=())],
            'output_toks': [len(ids) for ids in enc.encode_batch(assistant_msgs, disallowed_special=())],
            'num_turns': row['turn'],
        }

    processed = dataset.map(process_row, num_proc=64, remove_columns=dataset.column_names)
    stats = [row for row in processed if row['num_turns'] > 0]

    with open(stats_filepath, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f'Collected and saved stats to {stats_filepath}.')
    return stats


def filter_outliers(data, percentile):
    data = np.array(data)
    return data[data <= np.percentile(data, percentile)]


LOG_SCALE = False
STATS_FILEPATH = Path(__file__).resolve().parent / 'wildchat_stats.json'
stats = collect_stats(STATS_FILEPATH)

input_toks_all = np.array([tok for s in stats for tok in s['input_toks']])
output_toks_all = np.array([tok for s in stats for tok in s['output_toks']])
input_toks = filter_outliers(input_toks_all, percentile=85)
output_toks = filter_outliers(output_toks_all, percentile=95)
num_turns = filter_outliers([s['num_turns'] for s in stats], percentile=99)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Histogram of empirical input token counts — fit on full data
mu_i = np.mean(np.log(input_toks_all))
sigma_i = np.std(np.log(input_toks_all))
if LOG_SCALE:
    bins_i = np.logspace(np.log10(max(1, min(input_toks))), np.log10(max(input_toks)), 50)
    x_i = np.logspace(np.log10(max(1, min(input_toks))), np.log10(max(input_toks)), 200)
else:
    bins_i = 50
    x_i = np.linspace(1, max(input_toks), 200)
axes[0].hist(input_toks, bins=bins_i, edgecolor='black', density=True)
# Overlay fitted lognormal PDF
axes[0].plot(x_i, lognorm.pdf(x_i, s=sigma_i, scale=np.exp(mu_i)), 'r-', lw=2)
if LOG_SCALE:
    axes[0].set_xscale('log')
axes[0].set_title(f'WildChat - Input Tokens per Turn\nLogNormal(mu={mu_i:.4f}, sigma={sigma_i:.4f})')
axes[0].set_xlabel('Tokens')
axes[0].set_ylabel('Density')

# Histogram of empirical output token counts — fit on full data
mu_o = np.mean(np.log(output_toks_all))
sigma_o = np.std(np.log(output_toks_all))
if LOG_SCALE:
    bins_o = np.logspace(np.log10(max(1, min(output_toks))), np.log10(max(output_toks)), 50)
    x_o = np.logspace(np.log10(max(1, min(output_toks))), np.log10(max(output_toks)), 200)
else:
    bins_o = 50
    x_o = np.linspace(1, max(output_toks), 200)
axes[1].hist(output_toks, bins=bins_o, edgecolor='black', density=True)
# Overlay fitted lognormal PDF
axes[1].plot(x_o, lognorm.pdf(x_o, s=sigma_o, scale=np.exp(mu_o)), 'r-', lw=2)
if LOG_SCALE:
    axes[1].set_xscale('log')
axes[1].set_title(f'WildChat - Output Tokens per Turn\nLogNormal(mu={mu_o:.4f}, sigma={sigma_o:.4f})')
axes[1].set_xlabel('Tokens')

# Histogram of empirical turn counts per session
max_t = max(num_turns)
axes[2].hist(num_turns, bins=range(1, max_t + 2), edgecolor='black', align='left', density=True)
# Fit geometric distribution: p = 1/mean (MLE for geometric)
p = 1.0 / np.mean(num_turns)
x_t = np.arange(1, max_t + 1)
# Overlay fitted geometric PMF
axes[2].plot(x_t, geom.pmf(x_t, p), 'r-o', lw=2, markersize=4)
axes[2].set_title(f'WildChat - Turns per Session\nGeometric(p={p:.4f})')
axes[2].set_xlabel('Turns')

plt.tight_layout()
plt.savefig(Path(__file__).resolve().parent / 'wildchat_stats.png', dpi=150)
print('Saved wildchat_stats.png')

#!/usr/bin/env python3
# filepath: /Users/bzui/lsaie_project/plot_grad_clip.py
"""
Compare gradient clipping impact on DeepSpeed offload stages.
Focus: Throughput cost vs Loss improvement tradeoff.

Usage: python plot_grad_clip.py
"""

import re
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

LOG_DIR = Path("/Users/bzui/lsaie_project/logs/final")
OUTPUT_DIR = Path("/Users/bzui/lsaie_project")

STEP_PATTERN = re.compile(
    r"Step: (\d+) \| Loss: ([\d.]+) \| Tokens per second: ([\d.]+) \| "
    r"Training tokens per second \(%\): ([\d.]+) \| MFU \(%\): ([\d.]+) \| TFLOPs: ([\d.]+)"
)


def parse_log_file(filepath):
    data = {"steps": [], "loss": [], "tokens_per_sec": []}
    with open(filepath, 'r') as f:
        for match in STEP_PATTERN.finditer(f.read()):
            data["steps"].append(int(match.group(1)))
            data["loss"].append(float(match.group(2)))
            data["tokens_per_sec"].append(float(match.group(3)))
    return data


def aggregate_by_step(data):
    step_data = defaultdict(lambda: defaultdict(list))
    for i, step in enumerate(data["steps"]):
        step_data[step]["loss"].append(data["loss"][i])
        step_data[step]["tokens_per_sec"].append(data["tokens_per_sec"][i])
    
    aggregated = {"steps": sorted(step_data.keys()), "loss": [], "tokens_per_sec": []}
    for step in aggregated["steps"]:
        aggregated["loss"].append(np.mean(step_data[step]["loss"]))
        aggregated["tokens_per_sec"].append(np.sum(step_data[step]["tokens_per_sec"]))
    return aggregated


def load_grad_clip_logs():
    """Load logs for offload grad clip comparison only."""
    log_configs = {
        ("SuperOffload", "No Grad Clip"): "stage_superoffload_latest.out",
        ("SuperOffload", "With Grad Clip"): "stage_superoffload_grad_clip_latest.out",
        ("ZeroOffload", "No Grad Clip"): "stage_zerooffload_latest.out",
        ("ZeroOffload", "With Grad Clip"): "stage_zerooffload_grad_clip_latest.out",
    }
    
    all_data = {}
    for (stage, grad_clip), filename in log_configs.items():
        filepath = LOG_DIR / filename
        if filepath.exists():
            raw_data = parse_log_file(filepath)
            if raw_data["steps"]:
                all_data[(stage, grad_clip)] = aggregate_by_step(raw_data)
                print(f"✓ {stage} - {grad_clip}")
    return all_data


def compute_stats(all_data):
    """Compute summary statistics."""
    summary = {}
    for key, data in all_data.items():
        warmup = 10
        mask = [s >= warmup for s in data["steps"]]
        losses = [l for l, m in zip(data["loss"], mask) if m]
        throughputs = [t for t, m in zip(data["tokens_per_sec"], mask) if m]
        
        summary[key] = {
            "avg_throughput": np.mean(throughputs),
            "avg_loss": np.mean(losses),
            "final_loss": data["loss"][-1],
        }
    return summary


def plot_grad_clip_comparison(all_data, summary):
    """Create focused grad clip comparison: throughput vs loss tradeoff."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Gradient Clipping Tradeoff: Throughput vs Loss\n(Offload stages require rollback & recompute with grad clip)', 
                 fontsize=12, fontweight='bold')
    
    stages = ["ZeroOffload", "SuperOffload"]
    colors = {"With Grad Clip": "#e74c3c", "No Grad Clip": "#3498db"}
    
    for idx, stage in enumerate(stages):
        # Left column: Throughput over time
        ax_throughput = axes[idx, 0]
        for grad_clip in ["No Grad Clip", "With Grad Clip"]:
            key = (stage, grad_clip)
            if key in all_data:
                data = all_data[key]
                ax_throughput.plot(data["steps"], [t / 1000 for t in data["tokens_per_sec"]], 
                                  label=grad_clip, color=colors[grad_clip], linewidth=2, alpha=0.8)
        
        # Add throughput stats
        #no_clip_tp = summary.get((stage, "No Grad Clip"), {}).get("avg_throughput", 0) / 1000
        #with_clip_tp = summary.get((stage, "With Grad Clip"), {}).get("avg_throughput", 0) / 1000
        #slowdown = (1 - with_clip_tp / no_clip_tp) * 100 if no_clip_tp > 0 else 0
        
        ax_throughput.set_xlabel('Step')
        ax_throughput.set_ylabel('Throughput (k tokens/sec)')
        #ax_throughput.set_title(f'{stage} - Throughput\n(Grad clip: {slowdown:.1f}% slower)')
        ax_throughput.legend(fontsize=9)
        ax_throughput.grid(alpha=0.3)
        
        # Right column: Loss over time
        ax_loss = axes[idx, 1]
        for grad_clip in ["No Grad Clip", "With Grad Clip"]:
            key = (stage, grad_clip)
            if key in all_data:
                data = all_data[key]
                ax_loss.plot(data["steps"], data["loss"], 
                            label=grad_clip, color=colors[grad_clip], linewidth=2, alpha=0.8)
        
        # Add loss stats
        #no_clip_loss = summary.get((stage, "No Grad Clip"), {}).get("avg_loss", 0)
        #with_clip_loss = summary.get((stage, "With Grad Clip"), {}).get("avg_loss", 0)
        #loss_diff = with_clip_loss - no_clip_loss
        
        ax_loss.set_xlabel('Step')
        ax_loss.set_ylabel('Loss')
        #ax_loss.set_title(f'{stage} - Loss\n(Grad clip: {loss_diff:+.2f} avg loss)')
        ax_loss.legend(fontsize=9)
        ax_loss.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'grad_clip_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: grad_clip_comparison.png")
    plt.close()


def print_summary(summary):
    """Print summary table."""
    print("\n" + "="*75)
    print("GRADIENT CLIPPING: THROUGHPUT vs LOSS TRADEOFF")
    print("="*75)
    print(f"{'Stage':<15} {'Grad Clip':<15} {'Throughput (k/s)':>18} {'Avg Loss':>12}")
    print("-"*75)
    
    for stage in ["ZeroOffload", "SuperOffload"]:
        for grad_clip in ["No Grad Clip", "With Grad Clip"]:
            key = (stage, grad_clip)
            if key in summary:
                s = summary[key]
                print(f"{stage:<15} {grad_clip:<15} {s['avg_throughput']/1000:>16.2f} {s['avg_loss']:>12.2f}")
        
        # Print tradeoff
        no_clip = summary.get((stage, "No Grad Clip"), {})
        with_clip = summary.get((stage, "With Grad Clip"), {})
        if no_clip and with_clip:
            slowdown = (1 - with_clip['avg_throughput'] / no_clip['avg_throughput']) * 100
            loss_diff = with_clip['avg_loss'] - no_clip['avg_loss']
            print(f"{'':>30} Tradeoff: {slowdown:+.1f}% throughput, {loss_diff:+.2f} loss")
        print("-"*75)


def main():
    print("Loading offload grad clip logs...")
    print("="*60)
    
    all_data = load_grad_clip_logs()
    if not all_data:
        print("No valid log files found!")
        return
    
    summary = compute_stats(all_data)
    print_summary(summary)
    
    print("\nGenerating plot...")
    plot_grad_clip_comparison(all_data, summary)
    
    print("\nPlot saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
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

LOG_DIR = Path("/users/rasteiger/LSAIE-Project/logs")
OUTPUT_DIR = Path("/users/rasteiger/LSAIE-Project")

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
    stages = ["ZeroOffload", "SuperOffload"]
    colors = {"With Grad Clip": "#e74c3c", "No Grad Clip": "#3498db"}
    
    # Collect all data to determine shared y-axis limits
    all_throughputs = []
    all_losses = []
    for stage in stages:
        for grad_clip in ["No Grad Clip", "With Grad Clip"]:
            key = (stage, grad_clip)
            if key in all_data:
                data = all_data[key]
                all_throughputs.extend([t / 1000 for t in data["tokens_per_sec"]])
                all_losses.extend(data["loss"])
    
    # Calculate y-axis limits with some padding
    throughput_margin = (max(all_throughputs) - min(all_throughputs)) * 0.05
    loss_margin = (max(all_losses) - min(all_losses)) * 0.05
    throughput_ylim = (min(all_throughputs) - throughput_margin, max(all_throughputs) + throughput_margin)
    loss_ylim = (min(all_losses) - loss_margin, max(all_losses) + loss_margin)
    
    # Plot 1: Throughput comparison
    fig_throughput, axes_throughput = plt.subplots(1, 2, figsize=(16, 7))
    
    for idx, stage in enumerate(stages):
        ax = axes_throughput[idx]
        for grad_clip in ["No Grad Clip", "With Grad Clip"]:
            key = (stage, grad_clip)
            if key in all_data:
                data = all_data[key]
                ax.plot(data["steps"], [t / 1000 for t in data["tokens_per_sec"]], 
                       label=grad_clip, color=colors[grad_clip], linewidth=3, alpha=0.8)
        
        ax.set_xlabel('Step', fontsize=25)
        if idx == 0:
            ax.set_ylabel('Throughput (k tokens/sec)', fontsize=25)
        ax.set_ylim(throughput_ylim)
        ax.legend(fontsize=25, loc='best')
        ax.tick_params(axis='both', which='major', labelsize=25)
        ax.grid(alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Add stage titles
    for idx, stage in enumerate(stages):
        pos = axes_throughput[idx].get_position()
        center_x = (pos.x0 + pos.x1) / 2
        center_y = pos.y1 + 0.01
        fig_throughput.text(center_x, center_y, stage, ha='center', va='bottom', fontsize=30)
    
    plt.savefig(OUTPUT_DIR / 'grad_clip_throughput.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: grad_clip_throughput.png")
    plt.close()
    
    # Plot 2: Loss comparison
    fig_loss, axes_loss = plt.subplots(1, 2, figsize=(16, 7))
    
    for idx, stage in enumerate(stages):
        ax = axes_loss[idx]
        for grad_clip in ["No Grad Clip", "With Grad Clip"]:
            key = (stage, grad_clip)
            if key in all_data:
                data = all_data[key]
                ax.plot(data["steps"], data["loss"], 
                       label=grad_clip, color=colors[grad_clip], linewidth=3, alpha=0.8)
        
        ax.set_xlabel('Step', fontsize=25)
        if idx == 0:
            ax.set_ylabel('Loss', fontsize=25)
        ax.set_ylim(loss_ylim)
        ax.legend(fontsize=25, loc='best')
        ax.tick_params(axis='both', which='major', labelsize=25)
        ax.grid(alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Add stage titles
    for idx, stage in enumerate(stages):
        pos = axes_loss[idx].get_position()
        center_x = (pos.x0 + pos.x1) / 2
        center_y = pos.y1 + 0.01
        fig_loss.text(center_x, center_y, stage, ha='center', va='bottom', fontsize=30)
    
    plt.savefig(OUTPUT_DIR / 'grad_clip_loss.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: grad_clip_loss.png")
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
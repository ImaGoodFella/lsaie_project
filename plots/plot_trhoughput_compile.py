#!/usr/bin/env python3
"""
Combined Plot with Subplot Labels (a) and (b):
(a) Throughput: ZeroOffload & SuperOffload (Baseline vs DeepCompile)
(b) Loss: ZeroOffload (Baseline vs DeepCompile)

Usage: python plot_combined_metrics_labeled.py
"""

import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Update these paths if necessary
LOG_DIR = Path("/Users/bzui/lsaie_project/logs/final")
OUTPUT_DIR = Path("/Users/bzui/lsaie_project")

# Regex to capture metrics
TPS_PATTERN = re.compile(r"(?:Tokens per second|TPS): ([\d.]+)")
LOSS_PATTERN = re.compile(r"Loss: ([\d.]+)")
STEP_PATTERN = re.compile(r"Step: (\d+)")

def parse_log(filepath):
    """Parse log file for steps, loss, and throughput."""
    data = {"steps": [], "loss": [], "tps": []}
    
    if not filepath.exists():
        return data

    with open(filepath, 'r') as f:
        content = f.read()
        
        step_iter = STEP_PATTERN.finditer(content)
        loss_iter = LOSS_PATTERN.finditer(content)
        tps_iter = TPS_PATTERN.finditer(content)
        
        for step_m, loss_m, tps_m in zip(step_iter, loss_iter, tps_iter):
            data["steps"].append(int(step_m.group(1)))
            data["loss"].append(float(loss_m.group(1)))
            data["tps"].append(float(tps_m.group(1)))
            
    return data

def get_avg_tps(data, warmup=10):
    """Calculate average throughput excluding warmup."""
    if not data["tps"]:
        return 0.0
    
    valid_tps = [t for s, t in zip(data["steps"], data["tps"]) if s >= warmup]
    return np.mean(valid_tps) if valid_tps else 0.0

def load_all_data():
    """Load necessary data."""
    files = {
        "Zero_Base": LOG_DIR / "stage_zerooffload_latest.out",
        "Zero_Comp": LOG_DIR / "stage_zerooffload_deepcompile_latest.out",
        "Super_Base": LOG_DIR / "stage_superoffload_latest.out",
        "Super_Comp": LOG_DIR / "stage_superoffload_deepcompile_latest.out",
    }
    
    loaded = {}
    for key, filepath in files.items():
        loaded[key] = parse_log(filepath)
        print(f"{'✓' if loaded[key]['steps'] else '✗'} Loaded {key}")
        
    return loaded

def plot_combined(data):
    """Generate 1x2 plot with (a) and (b) labels."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # ==========================================
    # Plot (a): Throughput Bar Chart
    # ==========================================
    strategies = ["ZeroOffload", "SuperOffload"]
    baseline_vals = [
        get_avg_tps(data["Zero_Base"]),
        get_avg_tps(data["Super_Base"])
    ]
    compiled_vals = [
        get_avg_tps(data["Zero_Comp"]),
        get_avg_tps(data["Super_Comp"])
    ]
    
    x = np.arange(len(strategies))
    width = 0.35
    
    rects1 = ax1.bar(x - width/2, baseline_vals, width, label='Baseline', color='#95a5a6', edgecolor='black')
    rects2 = ax1.bar(x + width/2, compiled_vals, width, label='DeepCompile', color='#2ecc71', edgecolor='black')
    
    ax1.set_ylabel('Throughput (Tokens / Sec)', fontsize=12)
    ax1.set_title('Throughput Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies, fontsize=12)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, max(baseline_vals + compiled_vals) * 1.15)
    
    # Add values on bars
    for rect in rects1 + rects2:
        h = rect.get_height()
        if h > 0:
            ax1.text(rect.get_x() + rect.get_width()/2, h, f'{h:.0f}', 
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add label (a)
    ax1.text(-0.1, 1.05, '(a)', transform=ax1.transAxes, size=16, weight='bold')

    # ==========================================
    # Plot (b): Loss Line Chart (ZeroOffload only)
    # ==========================================
    ax2.plot(data["Zero_Base"]["steps"], data["Zero_Base"]["loss"], 
             label='Baseline', color='#95a5a6', linewidth=2, alpha=0.8)
    ax2.plot(data["Zero_Comp"]["steps"], data["Zero_Comp"]["loss"], 
             label='DeepCompile', color='#e74c3c', linewidth=2, alpha=0.9)
    
    ax2.set_xlabel('Training Steps', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Training Loss: ZeroOffload', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Add label (b)
    ax2.text(-0.1, 1.05, '(b)', transform=ax2.transAxes, size=16, weight='bold')

    plt.tight_layout()
    save_path = OUTPUT_DIR / 'combined_metrics_labeled.png'
    plt.savefig(save_path, dpi=150)
    print(f"\n✓ Combined plot saved to: {save_path}")

def main():
    print("Generating labeled figure...")
    print("="*50)
    data = load_all_data()
    plot_combined(data)

if __name__ == "__main__":
    main()
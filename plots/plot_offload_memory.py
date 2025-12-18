#!/usr/bin/env python3
"""
Compare Memory vs Throughput for DeepSpeed stages (Stage 3 + Offloads).
Plots Memory (Bar) and Throughput over Time (Line).

Usage: python plot_stage3_offload_metrics.py
"""

import re
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

LOG_DIR = Path("/Users/bzui/lsaie_project/logs/final")
OUTPUT_DIR = Path("/Users/bzui/lsaie_project")

# New format
STEP_PATTERN_NEW = re.compile(
    r"Step: (\d+) \| Loss: ([\d.]+) \| Tokens per second: ([\d.]+).*?"
    r"Mem Allocated \(GB\): ([\d.]+) \| Mem Reserved \(GB\): ([\d.]+) \| Max Mem Allocated \(GB\): ([\d.]+)"
)

# Old format
STEP_PATTERN_OLD = re.compile(
    r"Step: (\d+) \| Loss: ([\d.]+) \| TPS: ([\d.]+).*?"
    r"Mem: ([\d.]+)GB \| Reserved: ([\d.]+)GB \| Peak: ([\d.]+)GB"
)

# Fallback (throughput only)
STEP_PATTERN_NO_MEM = re.compile(
    r"Step: (\d+) \| Loss: ([\d.]+) \| Tokens per second: ([\d.]+)"
)

def parse_log_file(filepath):
    """Parse log file for memory and throughput data."""
    data = {
        "steps": [], 
        "tokens_per_sec": [],
        "mem_allocated": [], "mem_reserved": [], "mem_peak": [],
    }
    
    with open(filepath, 'r') as f:
        content = f.read()
        
        matches = list(STEP_PATTERN_NEW.finditer(content))
        if matches:
            for match in matches:
                data["steps"].append(int(match.group(1)))
                data["tokens_per_sec"].append(float(match.group(3)))
                data["mem_allocated"].append(float(match.group(4)))
                data["mem_reserved"].append(float(match.group(5)))
                data["mem_peak"].append(float(match.group(6)))
            return data
        
        matches = list(STEP_PATTERN_OLD.finditer(content))
        if matches:
            for match in matches:
                data["steps"].append(int(match.group(1)))
                data["tokens_per_sec"].append(float(match.group(3)))
                data["mem_allocated"].append(float(match.group(4)))
                data["mem_reserved"].append(float(match.group(5)))
                data["mem_peak"].append(float(match.group(6)))
            return data

        matches = list(STEP_PATTERN_NO_MEM.finditer(content))
        if matches:
            for match in matches:
                data["steps"].append(int(match.group(1)))
                data["tokens_per_sec"].append(float(match.group(3)))
    
    return data

def aggregate_by_step(data):
    """Aggregate data by step."""
    step_data = defaultdict(lambda: {"tps": [], "mem": [], "peak": []})
    
    for i, step in enumerate(data["steps"]):
        step_data[step]["tps"].append(data["tokens_per_sec"][i])
        if data["mem_allocated"]:
            step_data[step]["mem"].append(data["mem_allocated"][i])
            step_data[step]["peak"].append(data["mem_peak"][i])
    
    steps = sorted(step_data.keys())
    throughput = [np.mean(step_data[s]["tps"]) for s in steps]
    
    avg_mem = [np.mean(step_data[s]["mem"]) for s in steps] if data["mem_allocated"] else []
    avg_peak = [np.mean(step_data[s]["peak"]) for s in steps] if data["mem_peak"] else []
    
    return {
        "steps": steps,
        "throughput": throughput,
        "mem_allocated": avg_mem,
        "mem_peak": avg_peak,
    }

def load_all_logs():
    """Load logs and keep full time-series data."""
    log_configs = {
        "Stage 3": "stage_3_latest.out",
        "ZeroOffload": "stage_zerooffload_latest.out",
        "SuperOffload": "stage_superoffload_latest.out",
    }
    
    all_data = {}
    for stage, filename in log_configs.items():
        filepath = LOG_DIR / filename
        if filepath.exists():
            raw_data = parse_log_file(filepath)
            if raw_data["steps"]:
                agg = aggregate_by_step(raw_data)
                
                warmup = 10
                mask = [s >= warmup for s in agg["steps"]]
                
                # Scalar stats (for Bar Chart)
                if agg["mem_allocated"]:
                    scalar_avg_mem = np.mean([t for t, m in zip(agg["mem_allocated"], mask) if m])
                    scalar_peak_mem = np.max([t for t, m in zip(agg["mem_peak"], mask) if m])
                else:
                    scalar_avg_mem = scalar_peak_mem = 0
                
                # Store Time Series + Scalars
                all_data[stage] = {
                    "steps": agg["steps"],
                    "throughput_series": agg["throughput"],
                    "avg_mem": scalar_avg_mem,
                    "peak_mem": scalar_peak_mem,
                }
                print(f"✓ {stage}: Steps={len(agg['steps'])}, PeakMem={scalar_peak_mem:.1f}GB")
            else:
                print(f"✗ {stage}: No steps found")
        else:
            print(f"✗ {stage}: File not found")
    
    return all_data

def plot_metrics(all_data):
    """Create 2-row plot: Top=Memory (Bar), Bottom=Throughput (Line)."""
    fig, (ax_mem, ax_tput) = plt.subplots(2, 1, figsize=(10, 10))
    fig.suptitle('DeepSpeed Stage 3 & Offloading Strategies', fontsize=16, fontweight='bold')
    
    stages_order = ["Stage 3", "ZeroOffload", "SuperOffload"]
    stages = [s for s in stages_order if s in all_data]
    
    # ===== 1. Memory Plot (Bar Chart) =====
    x = np.arange(len(stages))
    width = 0.35
    
    peak_mem = [all_data.get(s, {}).get("peak_mem", 0) for s in stages]
    avg_mem = [all_data.get(s, {}).get("avg_mem", 0) for s in stages]
    
    bars1 = ax_mem.bar(x - width/2, peak_mem, width, label='Peak Memory', color='#e74c3c', edgecolor='black', alpha=0.9)
    bars2 = ax_mem.bar(x + width/2, avg_mem, width, label='Avg Memory', color='#3498db', edgecolor='black', alpha=0.9)
    
    ax_mem.set_ylabel('GPU Memory (GB)', fontsize=12)
    ax_mem.set_title('GPU Memory Usage', fontsize=12)
    ax_mem.set_xticks(x)
    ax_mem.set_xticklabels(stages)
    ax_mem.legend(loc='upper right')
    ax_mem.grid(axis='y', alpha=0.3, linestyle='--')
    ax_mem.set_ylim(0, 85)
    
    for bar in bars1 + bars2:
        h = bar.get_height()
        if h > 0:
            ax_mem.text(bar.get_x() + bar.get_width()/2, h + 1, f'{h:.1f}', ha='center', va='bottom', fontsize=10)

    # ===== 2. Throughput Plot (Line Chart) =====
    # Colors for lines
    colors = {
        "Stage 3": "#f39c12",       # Orange
        "ZeroOffload": "#27ae60",   # Green
        "SuperOffload": "#8e44ad"   # Purple
    }
    
    for stage in stages:
        data = all_data[stage]
        steps = data["steps"]
        tput = data["throughput_series"]
        
        ax_tput.plot(steps, tput, label=stage, color=colors.get(stage, 'black'), linewidth=2)
    
    ax_tput.set_xlabel('Training Steps', fontsize=12)
    ax_tput.set_ylabel('Throughput (tokens/sec)', fontsize=12)
    ax_tput.set_title('Training Throughput Over Time', fontsize=12)
    ax_tput.legend(loc='lower right')
    ax_tput.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'stage3_offload_metrics_time.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: stage3_offload_metrics_time.png")
    plt.close()

def main():
    print("Loading logs...")
    print("="*60)
    
    all_data = load_all_logs()
    if not all_data:
        print("No valid log files found!")
        return
    
    print("\nGenerating plot...")
    plot_metrics(all_data)
    
    print("\nPlot saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
############################################################
#
# Neural Compiler: Compiling scheme into composable and
#                  differentiable neural network representations
#
# plot_benchmarks.py: Generates plots from benchmark CSV data showing performance scaling
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Generate paper-ready plots from benchmark CSV data.

Usage:
    python -m benchmarks.plot_benchmarks benchmarks/results.csv
    python -m benchmarks.plot_benchmarks benchmarks/results.csv --output-dir figures/
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


COLORS = {
    "python": "#8BC34A",
    "guile": "#795548",
    "sequential": "#4CAF50",
    "SchemeGNN": "#2196F3",
    "Direct_CPU": "#FF9800",
    "Direct_GPU": "#FF5722",
    "batch_cpu": "#2196F3",
    "batch_gpu": "#FF5722",
    "numpy": "#8BC34A",
    "torch_cpu": "#FF9800",
    "torch_gpu": "#E91E63",
}


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in ["mean_s", "std_s", "min_s", "median_s"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["throughput"] = pd.to_numeric(df["throughput"], errors="coerce")
    return df


def _save(fig, output_dir, name):
    for ext in ("pdf", "png"):
        path = os.path.join(output_dir, f"{name}.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"  Saved {output_dir}/{name}.{{pdf,png}}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1: Single-input evaluation — all evaluators + baselines
# ---------------------------------------------------------------------------

def plot_evaluator_comparison(df: pd.DataFrame, output_dir: str):
    eval_df = df[df["benchmark"] == "evaluator"].copy()
    baseline_df = df[(df["benchmark"] == "baseline_single")].copy()
    combined = pd.concat([eval_df, baseline_df], ignore_index=True)
    if combined.empty:
        return

    programs = eval_df["program"].unique() if not eval_df.empty else baseline_df["program"].unique()
    order = ["python", "guile", "sequential", "SchemeGNN", "Direct_CPU", "Direct_GPU"]
    evaluators = [e for e in order if e in combined["evaluator"].values]

    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = np.arange(len(programs))
    width = 0.8 / len(evaluators)

    for i, ev in enumerate(evaluators):
        ev_data = combined[combined["evaluator"] == ev]
        times = []
        for prog in programs:
            row = ev_data[ev_data["program"] == prog]
            times.append(row["mean_s"].values[0] if len(row) > 0 else np.nan)
        offset = (i - len(evaluators) / 2 + 0.5) * width
        ax.bar(
            x + offset, times, width,
            label=ev, color=COLORS.get(ev, "gray"), alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(programs, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Time per evaluation (seconds)")
    ax.set_yscale("log")
    ax.set_title("Single-Input Evaluation Time")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, output_dir, "evaluator_comparison")


# ---------------------------------------------------------------------------
# Figure 2: Single-input throughput bar chart
# ---------------------------------------------------------------------------

def plot_evaluator_throughput(df: pd.DataFrame, output_dir: str):
    eval_df = df[df["benchmark"] == "evaluator"].copy()
    baseline_df = df[df["benchmark"] == "baseline_single"].copy()
    combined = pd.concat([eval_df, baseline_df], ignore_index=True)
    if combined.empty:
        return

    programs = eval_df["program"].unique() if not eval_df.empty else baseline_df["program"].unique()
    order = ["python", "guile", "sequential", "SchemeGNN", "Direct_CPU", "Direct_GPU"]
    evaluators = [e for e in order if e in combined["evaluator"].values]

    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = np.arange(len(programs))
    width = 0.8 / len(evaluators)

    for i, ev in enumerate(evaluators):
        ev_data = combined[combined["evaluator"] == ev]
        tps = []
        for prog in programs:
            row = ev_data[ev_data["program"] == prog]
            tps.append(row["throughput"].values[0] if len(row) > 0 else np.nan)
        offset = (i - len(evaluators) / 2 + 0.5) * width
        ax.bar(
            x + offset, tps, width,
            label=ev, color=COLORS.get(ev, "gray"), alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(programs, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Throughput (evals/s)")
    ax.set_yscale("log")
    ax.set_title("Single-Input Throughput (higher is better)")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, output_dir, "evaluator_throughput")


# ---------------------------------------------------------------------------
# Figure 3: Batch throughput — GNN vs native baselines
# ---------------------------------------------------------------------------

def plot_batch_vs_native(df: pd.DataFrame, output_dir: str):
    batch_df = df[df["benchmark"] == "batch_scaling"].copy()
    baseline_df = df[df["benchmark"] == "baseline_batch"].copy()
    if batch_df.empty:
        return

    programs_to_show = ["add", "four_ops", "discriminant", "dot4"]
    programs_to_show = [p for p in programs_to_show if p in batch_df["program"].values]
    if not programs_to_show:
        programs_to_show = list(batch_df["program"].unique()[:4])

    all_evaluators = ["numpy", "torch_cpu", "torch_gpu", "batch_cpu", "batch_gpu"]
    labels = {
        "numpy": "NumPy",
        "torch_cpu": "PyTorch CPU",
        "torch_gpu": "PyTorch GPU",
        "batch_cpu": "GNN batch CPU",
        "batch_gpu": "GNN batch GPU",
    }
    markers = {"numpy": "^", "torch_cpu": "v", "torch_gpu": "D", "batch_cpu": "o", "batch_gpu": "s"}
    linestyles = {"numpy": "--", "torch_cpu": "--", "torch_gpu": "--", "batch_cpu": "-", "batch_gpu": "-"}

    combined = pd.concat([batch_df, baseline_df], ignore_index=True)

    fig, axes = plt.subplots(
        1, len(programs_to_show),
        figsize=(4.2 * len(programs_to_show), 4),
        squeeze=False,
    )

    for idx, prog in enumerate(programs_to_show):
        ax = axes[0, idx]
        prog_df = combined[combined["program"] == prog]

        for ev in all_evaluators:
            ev_df = prog_df[prog_df["evaluator"] == ev].sort_values("batch_size")
            if ev_df.empty:
                continue
            ax.plot(
                ev_df["batch_size"], ev_df["throughput"],
                marker=markers.get(ev, "o"), linestyle=linestyles.get(ev, "-"),
                color=COLORS.get(ev, "gray"),
                label=labels.get(ev, ev), markersize=4, linewidth=1.5,
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Batch size")
        if idx == 0:
            ax.set_ylabel("Throughput (evals/s)")
        ax.set_title(prog, fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Batch Throughput: GNN vs. Native Baselines", fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, output_dir, "batch_vs_native")


# ---------------------------------------------------------------------------
# Figure 4: Batch scaling (GNN only, CPU vs GPU)
# ---------------------------------------------------------------------------

def plot_batch_scaling(df: pd.DataFrame, output_dir: str):
    batch_df = df[df["benchmark"] == "batch_scaling"].copy()
    if batch_df.empty:
        return

    programs = batch_df["program"].unique()
    evaluators = batch_df["evaluator"].unique()

    fig, axes = plt.subplots(
        1, min(len(programs), 4),
        figsize=(4 * min(len(programs), 4), 3.5),
        squeeze=False,
    )

    labels = {"batch_cpu": "CPU", "batch_gpu": "GPU"}

    for idx, prog in enumerate(programs[:4]):
        ax = axes[0, idx]
        prog_df = batch_df[batch_df["program"] == prog]

        for ev in evaluators:
            ev_df = prog_df[prog_df["evaluator"] == ev].sort_values("batch_size")
            if ev_df.empty:
                continue
            ax.plot(
                ev_df["batch_size"], ev_df["throughput"],
                "o-", color=COLORS.get(ev, "gray"),
                label=labels.get(ev, ev), markersize=4, linewidth=1.5,
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Batch size")
        if idx == 0:
            ax.set_ylabel("Throughput (evals/s)")
        ax.set_title(prog, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("GNN Batch Throughput Scaling", fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, output_dir, "batch_scaling")


# ---------------------------------------------------------------------------
# Figure 5: Overhead ratio — GNN sequential vs baselines
# ---------------------------------------------------------------------------

def plot_overhead_ratio(df: pd.DataFrame, output_dir: str):
    eval_df = df[df["benchmark"] == "evaluator"].copy()
    baseline_df = df[df["benchmark"] == "baseline_single"].copy()
    if eval_df.empty or baseline_df.empty:
        return

    programs = eval_df["program"].unique()

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(programs))

    for ref_name, ref_label, color in [
        ("python", "vs Python native", "#8BC34A"),
        ("guile", "vs Guile Scheme", "#795548"),
    ]:
        ratios = []
        for prog in programs:
            seq_row = eval_df[(eval_df["program"] == prog) & (eval_df["evaluator"] == "sequential")]
            ref_row = baseline_df[(baseline_df["program"] == prog) & (baseline_df["evaluator"] == ref_name)]
            if len(seq_row) > 0 and len(ref_row) > 0:
                ratio = seq_row["mean_s"].values[0] / ref_row["mean_s"].values[0]
                ratios.append(ratio)
            else:
                ratios.append(np.nan)

        offset = -0.2 if ref_name == "python" else 0.2
        ax.bar(x + offset, ratios, 0.35, label=ref_label, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(programs, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Slowdown factor (sequential / baseline)")
    ax.set_title("GNN Sequential Evaluator Overhead vs. Native Execution")
    ax.legend(fontsize=9)
    ax.axhline(y=1, color="black", linestyle="--", alpha=0.3)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, output_dir, "overhead_ratio")


# ---------------------------------------------------------------------------
# Figure 6: Depth and width scaling
# ---------------------------------------------------------------------------

def plot_complexity_scaling(df: pd.DataFrame, output_dir: str):
    depth_df = df[df["benchmark"] == "depth_scaling"].copy()
    width_df = df[df["benchmark"] == "width_scaling"].copy()
    if depth_df.empty and width_df.empty:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    labels = {"batch_cpu": "CPU", "batch_gpu": "GPU"}

    if not depth_df.empty:
        for ev in depth_df["evaluator"].unique():
            ev_data = depth_df[depth_df["evaluator"] == ev].sort_values("depth")
            ax1.plot(
                ev_data["depth"], ev_data["mean_s"] * 1000,
                "o-", color=COLORS.get(ev, "gray"),
                label=labels.get(ev, ev), markersize=4, linewidth=1.5,
            )
        ax1.set_xlabel("DAG depth (message passing rounds)")
        ax1.set_ylabel("Time per batch (ms)")
        ax1.set_title("Depth Scaling (batch=1000)")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

    if not width_df.empty:
        for ev in width_df["evaluator"].unique():
            ev_data = width_df[width_df["evaluator"] == ev].sort_values("nodes")
            ax2.plot(
                ev_data["nodes"], ev_data["mean_s"] * 1000,
                "o-", color=COLORS.get(ev, "gray"),
                label=labels.get(ev, ev), markersize=4, linewidth=1.5,
            )
        ax2.set_xlabel("Node count")
        ax2.set_ylabel("Time per batch (ms)")
        ax2.set_title("Width Scaling (batch=1000)")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, output_dir, "complexity_scaling")


# ---------------------------------------------------------------------------
# Figure 7: Compilation time
# ---------------------------------------------------------------------------

def plot_compilation(df: pd.DataFrame, output_dir: str):
    comp_df = df[df["benchmark"] == "compilation"].copy()
    if comp_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 3.5))
    programs = comp_df["program"].unique()
    x = np.arange(len(programs))

    compile_times = []
    init_times = []
    for prog in programs:
        ct = comp_df[(comp_df["program"] == prog) & (comp_df["evaluator"] == "compile")]
        compile_times.append(ct["mean_s"].values[0] * 1000 if len(ct) > 0 else 0)
        mt = comp_df[(comp_df["program"] == prog) & (comp_df["evaluator"] == "model_init")]
        init_times.append(mt["mean_s"].values[0] * 1000 if len(mt) > 0 else 0)

    width = 0.35
    ax.bar(x - width/2, compile_times, width, label="Scheme → Graph", color="#4CAF50")
    ax.bar(x + width/2, init_times, width, label="Graph → DirectModule", color="#2196F3")

    ax.set_xticks(x)
    ax.set_xticklabels(programs, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Time (ms)")
    ax.set_title("One-Time Compilation Cost")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, output_dir, "compilation_time")


# ---------------------------------------------------------------------------
# Figure 8: Large programs — throughput comparison
# ---------------------------------------------------------------------------

def plot_large_programs(df: pd.DataFrame, output_dir: str):
    lp_df = df[df["benchmark"] == "large_program"].copy()
    if lp_df.empty:
        return

    tree_df = lp_df[lp_df["program"].str.startswith("tree_")]
    chain_df = lp_df[lp_df["program"].str.startswith("chain_")]

    all_evals = ["python_scalar", "numpy", "torch_cpu", "torch_gpu",
                 "batch_cpu", "batch_gpu"]
    labels = {
        "python_scalar": "Python scalar",
        "numpy": "NumPy vectorized",
        "torch_cpu": "PyTorch CPU",
        "torch_gpu": "PyTorch GPU",
        "batch_cpu": "GNN batch CPU",
        "batch_gpu": "GNN batch GPU",
    }
    colors = {
        "python_scalar": "#8BC34A",
        "numpy": "#FFC107",
        "torch_cpu": "#FF9800",
        "torch_gpu": "#FF5722",
        "batch_cpu": "#2196F3",
        "batch_gpu": "#9C27B0",
    }
    markers = {
        "python_scalar": "^",
        "numpy": "v",
        "torch_cpu": "D",
        "torch_gpu": "d",
        "batch_cpu": "o",
        "batch_gpu": "s",
    }
    linestyles = {
        "python_scalar": ":",
        "numpy": "--",
        "torch_cpu": "--",
        "torch_gpu": "--",
        "batch_cpu": "-",
        "batch_gpu": "-",
    }

    has_trees = not tree_df.empty
    has_chains = not chain_df.empty
    ncols = int(has_trees) + int(has_chains)
    if ncols == 0:
        return

    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5), squeeze=False)
    col = 0

    if has_trees:
        ax = axes[0, col]
        col += 1

        def _extract_size(name):
            return int(name.replace("tree_", ""))

        for ev in all_evals:
            ev_df = tree_df[tree_df["evaluator"] == ev].copy()
            if ev_df.empty:
                continue
            ev_df["size"] = ev_df["program"].apply(_extract_size)
            ev_df = ev_df.sort_values("size")
            ax.plot(
                ev_df["size"], ev_df["throughput"],
                marker=markers.get(ev, "o"),
                linestyle=linestyles.get(ev, "-"),
                color=colors.get(ev, "gray"),
                label=labels.get(ev, ev),
                markersize=5, linewidth=1.5,
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Tree width (number of leaf multiplications)")
        ax.set_ylabel("Throughput (evals/s)")
        ax.set_title("Wide Trees (depth = log₂(width))")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    if has_chains:
        ax = axes[0, col]

        def _extract_depth(name):
            return int(name.replace("chain_", ""))

        for ev in all_evals:
            ev_df = chain_df[chain_df["evaluator"] == ev].copy()
            if ev_df.empty:
                continue
            ev_df["size"] = ev_df["program"].apply(_extract_depth)
            ev_df = ev_df.sort_values("size")
            ax.plot(
                ev_df["size"], ev_df["throughput"],
                marker=markers.get(ev, "o"),
                linestyle=linestyles.get(ev, "-"),
                color=colors.get(ev, "gray"),
                label=labels.get(ev, ev),
                markersize=5, linewidth=1.5,
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Chain depth (number of add operations)")
        ax.set_ylabel("Throughput (evals/s)")
        ax.set_title("Deep Chains (depth = node count)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Large Program Throughput: GNN Batch vs. Native Baselines", fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, output_dir, "large_programs")


# ---------------------------------------------------------------------------
# Figure 9: GNN speedup over Python scalar for large programs
# ---------------------------------------------------------------------------

def plot_gnn_speedup(df: pd.DataFrame, output_dir: str):
    lp_df = df[df["benchmark"] == "large_program"].copy()
    if lp_df.empty:
        return

    tree_df = lp_df[lp_df["program"].str.startswith("tree_")]
    chain_df = lp_df[lp_df["program"].str.startswith("chain_")]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), squeeze=False)

    for col, (sub_df, kind, xlabel) in enumerate([
        (tree_df, "tree_", "Tree width"),
        (chain_df, "chain_", "Chain depth"),
    ]):
        if sub_df.empty:
            continue
        ax = axes[0, col]

        programs = sorted(sub_df["program"].unique(),
                          key=lambda p: int(p.replace(kind, "")))
        sizes = [int(p.replace(kind, "")) for p in programs]

        for gnn_ev, gnn_label, color in [
            ("batch_cpu", "GNN batch CPU", "#2196F3"),
            ("batch_gpu", "GNN batch GPU", "#9C27B0"),
        ]:
            speedups = []
            valid_sizes = []
            for prog, sz in zip(programs, sizes):
                py_row = sub_df[(sub_df["program"] == prog) & (sub_df["evaluator"] == "python_scalar")]
                gnn_row = sub_df[(sub_df["program"] == prog) & (sub_df["evaluator"] == gnn_ev)]
                if len(py_row) > 0 and len(gnn_row) > 0:
                    py_tp = py_row["throughput"].values[0]
                    gnn_tp = gnn_row["throughput"].values[0]
                    if py_tp > 0:
                        speedups.append(gnn_tp / py_tp)
                        valid_sizes.append(sz)

            if speedups:
                ax.plot(valid_sizes, speedups, "o-", color=color,
                        label=gnn_label, markersize=5, linewidth=1.5)

        ax.axhline(y=1, color="black", linestyle="--", alpha=0.3, label="Break-even")
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Speedup vs. Python scalar")
        ax.set_title(f"GNN Batch Speedup ({kind.replace('_', '').title()}s)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, output_dir, "gnn_speedup")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results")
    parser.add_argument("csv", help="Path to benchmark CSV file")
    parser.add_argument("--output-dir", default="benchmarks/figures", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = load_data(args.csv)

    print(f"Loaded {len(df)} rows from {args.csv}")
    print(f"Benchmarks: {df['benchmark'].unique().tolist()}")
    print(f"Generating figures in {args.output_dir}/")

    plot_evaluator_comparison(df, args.output_dir)
    plot_evaluator_throughput(df, args.output_dir)
    plot_batch_vs_native(df, args.output_dir)
    plot_batch_scaling(df, args.output_dir)
    plot_overhead_ratio(df, args.output_dir)
    plot_complexity_scaling(df, args.output_dir)
    plot_compilation(df, args.output_dir)
    plot_large_programs(df, args.output_dir)
    plot_gnn_speedup(df, args.output_dir)

    print("Done.")


if __name__ == "__main__":
    main()

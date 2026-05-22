############################################################
#
# Neural Compiler: Compiling scheme into composable and
#                  differentiable neural network representations
#
# compositional_generalization.py: Zero-error composition of compiled modules vs neural approximation error accumulation
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Compositional generalization: exact compiled chains vs neural approximation chains.

Demonstrates that compiled modules compose with zero error at any depth,
while neural approximations accumulate errors that grow exponentially with
composition depth. This answers "why compile?" -- because exact modules
compose exactly, approximate modules don't.

Eight modules are each compiled as frozen compiled modules AND approximated by
trained MLPs. Chains of 2-6 modules are evaluated with both representations.
Compiled chains match ground truth to float precision at all depths; neural
chains degrade measurably by depth 3 and catastrophically by depth 5-6.
"""

import argparse
import math
import pickle
import sys
import os
import time
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_compiler.compiler import compile_scheme
from neural_compiler.evaluator import DirectModule


# ---------------------------------------------------------------------------
# Module definitions
# ---------------------------------------------------------------------------

MODULE_SPECS = [
    ("square",   "(* x x)"),
    ("cube",     "(let ((x2 (* x x))) (* x x2))"),
    ("sin",      "(sin x)"),
    ("exp",      "(exp x)"),
    ("add_one",  "(+ x 1)"),
    ("negate",   "(- 0 x)"),
    ("double",   "(* 2 x)"),
    ("sqrt_abs", "(sqrt (+ (* x x) 0.01))"),
]


CHAIN_SPECS = [
    # 2-stage chains
    ("square -> add_one",                    ["square", "add_one"]),
    ("sin -> square",                        ["sin", "square"]),
    # 3-stage chains
    ("square -> add_one -> cube",            ["square", "add_one", "cube"]),
    ("exp -> negate -> add_one",             ["exp", "negate", "add_one"]),
    # 4-stage chains
    ("sin -> square -> add_one -> sqrt_abs", ["sin", "square", "add_one", "sqrt_abs"]),
    ("square -> double -> sin -> add_one",   ["square", "double", "sin", "add_one"]),
    # 5-stage chains
    ("sin -> exp -> negate -> add_one -> square",
     ["sin", "exp", "negate", "add_one", "square"]),
    ("square -> add_one -> cube -> negate -> add_one",
     ["square", "add_one", "cube", "negate", "add_one"]),
    # 6-stage chains
    ("negate -> add_one -> square -> double -> sin -> add_one",
     ["negate", "add_one", "square", "double", "sin", "add_one"]),
]


def ground_truth_module(name: str, x: torch.Tensor) -> torch.Tensor:
    if name == "square":
        return x * x
    elif name == "cube":
        return x * x * x
    elif name == "sin":
        return torch.sin(x)
    elif name == "exp":
        return torch.exp(x)
    elif name == "add_one":
        return x + 1
    elif name == "negate":
        return -x
    elif name == "double":
        return 2 * x
    elif name == "sqrt_abs":
        return torch.sqrt(x * x + 0.01)
    raise ValueError(f"Unknown module: {name}")


def ground_truth_chain(names: list[str], x: torch.Tensor) -> torch.Tensor:
    z = x
    for name in names:
        z = ground_truth_module(name, z)
    return z


# ---------------------------------------------------------------------------
# Neural approximation modules
# ---------------------------------------------------------------------------

class NeuralModule(nn.Module):
    def __init__(self, hidden: int = 32, layers: int = 2):
        super().__init__()
        mods = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(layers - 1):
            mods.extend([nn.Linear(hidden, hidden), nn.Tanh()])
        mods.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            return self.net(x.unsqueeze(1)).squeeze(1)
        return self.net(x).squeeze(-1)


def train_neural_module(name: str, hidden: int = 32, layers: int = 2,
                        n_samples: int = 10000, epochs: int = 5000,
                        lr: float = 1e-3, train_range: float = 2.0,
                        seed: int = 42) -> NeuralModule:
    torch.manual_seed(seed)
    model = NeuralModule(hidden=hidden, layers=layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    x_train = torch.rand(n_samples) * 2 * train_range - train_range
    y_train = ground_truth_module(name, x_train)

    for epoch in range(epochs):
        pred = model(x_train)
        loss = F.mse_loss(pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    with torch.no_grad():
        final_mse = F.mse_loss(model(x_train), y_train).item()

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return model, final_mse


# ---------------------------------------------------------------------------
# Chain evaluation
# ---------------------------------------------------------------------------

def eval_compiled_chain(subgraphs: list[DirectModule], x: torch.Tensor) -> torch.Tensor:
    z = x
    for sg in subgraphs:
        z = sg.forward_batch({"x": z})
    return z


def eval_neural_chain(modules: list[NeuralModule], x: torch.Tensor) -> torch.Tensor:
    z = x
    for m in modules:
        z = m(z)
    return z


def eval_handcoded_chain(module_names: list[str], x: torch.Tensor) -> torch.Tensor:
    """Compose hand-coded PyTorch functions — same math as ground_truth_chain."""
    z = x
    for name in module_names:
        z = ground_truth_module(name, z)
    return z


@dataclass
class ChainResult:
    name: str
    depth: int
    compiled_mse: float = 0.0
    neural_mse: float = 0.0
    handcoded_mse: float = 0.0
    compiled_max_err: float = 0.0
    neural_max_err: float = 0.0
    handcoded_max_err: float = 0.0
    compiled_mse_2x: float = 0.0
    neural_mse_2x: float = 0.0
    handcoded_mse_2x: float = 0.0
    compiled_max_err_2x: float = 0.0
    neural_max_err_2x: float = 0.0
    handcoded_max_err_2x: float = 0.0
    compiled_mse_4x: float = 0.0
    neural_mse_4x: float = 0.0
    handcoded_mse_4x: float = 0.0
    compiled_max_err_4x: float = 0.0
    neural_max_err_4x: float = 0.0
    handcoded_max_err_4x: float = 0.0


def evaluate_chain(chain_name: str, module_names: list[str],
                   compiled_modules: dict[str, DirectModule],
                   neural_modules: dict[str, NeuralModule],
                   n_eval: int = 10000, train_range: float = 2.0) -> ChainResult:
    result = ChainResult(name=chain_name, depth=len(module_names))
    compiled_chain = [compiled_modules[n] for n in module_names]
    neural_chain = [neural_modules[n] for n in module_names]

    for mult, suffix in [(1.0, ""), (2.0, "_2x"), (4.0, "_4x")]:
        r = train_range * mult
        x = torch.linspace(-r, r, n_eval)
        gt = ground_truth_chain(module_names, x)

        with torch.no_grad():
            try:
                c_pred = eval_compiled_chain(compiled_chain, x)
                c_mse = F.mse_loss(c_pred, gt).item()
                c_max = (c_pred - gt).abs().max().item()
            except Exception:
                c_mse, c_max = float("inf"), float("inf")

            try:
                n_pred = eval_neural_chain(neural_chain, x)
                n_mse = F.mse_loss(n_pred, gt).item()
                n_max = (n_pred - gt).abs().max().item()
            except Exception:
                n_mse, n_max = float("inf"), float("inf")

            try:
                hc_pred = eval_handcoded_chain(module_names, x)
                hc_mse = F.mse_loss(hc_pred, gt).item()
                hc_max = (hc_pred - gt).abs().max().item()
            except Exception:
                hc_mse, hc_max = float("inf"), float("inf")

        setattr(result, f"compiled_mse{suffix}", c_mse)
        setattr(result, f"compiled_max_err{suffix}", c_max)
        setattr(result, f"neural_mse{suffix}", n_mse)
        setattr(result, f"neural_max_err{suffix}", n_max)
        setattr(result, f"handcoded_mse{suffix}", hc_mse)
        setattr(result, f"handcoded_max_err{suffix}", hc_max)

    return result


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize(results: list[ChainResult], module_errors: dict[str, float],
              save_path: str):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Experiment 3A: Exact Composition vs Neural Approximation",
                 fontsize=16, fontweight="bold")

    depths = sorted(set(r.depth for r in results))

    # Panel 1: MSE vs depth (in-distribution)
    ax = axes[0, 0]
    for depth in depths:
        rs = [r for r in results if r.depth == depth]
        c_vals = [r.compiled_mse for r in rs]
        hc_vals = [r.handcoded_mse for r in rs]
        n_vals = [r.neural_mse for r in rs]
        ax.scatter([depth - 0.1] * len(c_vals), [max(v, 1e-16) for v in c_vals],
                   c="tab:blue", marker="o", s=40, zorder=5)
        ax.scatter([depth] * len(hc_vals), [max(v, 1e-16) for v in hc_vals],
                   c="tab:green", marker="^", s=40, zorder=5)
        ax.scatter([depth + 0.1] * len(n_vals), [max(v, 1e-16) for v in n_vals],
                   c="tab:red", marker="s", s=40, zorder=5)

    ax.scatter([], [], c="tab:blue", marker="o", label="Compiled")
    ax.scatter([], [], c="tab:green", marker="^", label="Hand-coded")
    ax.scatter([], [], c="tab:red", marker="s", label="Neural")
    ax.set_yscale("log")
    ax.set_xlabel("Chain depth")
    ax.set_ylabel("MSE (in-distribution)")
    ax.set_title("Composition Error vs Depth")
    ax.legend()
    ax.set_xticks(depths)

    # Panel 2: MSE vs depth (4x extrapolation)
    ax = axes[0, 1]
    for depth in depths:
        rs = [r for r in results if r.depth == depth]
        c_vals = [r.compiled_mse_4x for r in rs]
        hc_vals = [r.handcoded_mse_4x for r in rs]
        n_vals = [r.neural_mse_4x for r in rs]
        ax.scatter([depth - 0.1] * len(c_vals), [max(v, 1e-16) for v in c_vals],
                   c="tab:blue", marker="o", s=40, zorder=5)
        ax.scatter([depth] * len(hc_vals), [max(v, 1e-16) for v in hc_vals],
                   c="tab:green", marker="^", s=40, zorder=5)
        ax.scatter([depth + 0.1] * len(n_vals), [max(v, 1e-16) for v in n_vals],
                   c="tab:red", marker="s", s=40, zorder=5)

    ax.scatter([], [], c="tab:blue", marker="o", label="Compiled")
    ax.scatter([], [], c="tab:green", marker="^", label="Hand-coded")
    ax.scatter([], [], c="tab:red", marker="s", label="Neural")
    ax.set_yscale("log")
    ax.set_xlabel("Chain depth")
    ax.set_ylabel("MSE (4x extrapolation)")
    ax.set_title("Extrapolation Error vs Depth")
    ax.legend()
    ax.set_xticks(depths)

    # Panel 3: Max absolute error comparison
    ax = axes[0, 2]
    names = [r.name for r in results]
    c_max = [max(r.compiled_max_err, 1e-16) for r in results]
    hc_max = [max(r.handcoded_max_err, 1e-16) for r in results]
    n_max = [max(r.neural_max_err, 1e-16) for r in results]
    x_pos = range(len(results))
    bar_w = 0.25
    ax.bar([p - bar_w for p in x_pos], c_max, bar_w, label="Compiled", color="tab:blue")
    ax.bar(list(x_pos), hc_max, bar_w, label="Hand-coded", color="tab:green")
    ax.bar([p + bar_w for p in x_pos], n_max, bar_w, label="Neural", color="tab:red")
    ax.set_yscale("log")
    ax.set_ylabel("Max absolute error")
    ax.set_title("Worst-Case Error (in-dist)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"d={r.depth}" for r in results], rotation=45, ha="right", fontsize=7)
    ax.legend()

    # Panel 4: Per-module neural approximation error
    ax = axes[1, 0]
    mod_names = list(module_errors.keys())
    mod_errs = list(module_errors.values())
    ax.bar(range(len(mod_names)), [max(e, 1e-16) for e in mod_errs], color="tab:orange")
    ax.set_yscale("log")
    ax.set_ylabel("Training MSE")
    ax.set_title("Per-Module Neural Approximation Error")
    ax.set_xticks(range(len(mod_names)))
    ax.set_xticklabels(mod_names, rotation=45, ha="right")

    # Panel 5: Neural/Compiled error ratio vs depth
    ax = axes[1, 1]
    for r in results:
        ratio_in = r.neural_mse / max(r.compiled_mse, 1e-16)
        ratio_4x = r.neural_mse_4x / max(r.compiled_mse_4x, 1e-16)
        ax.scatter(r.depth, ratio_in, c="tab:green", marker="o", s=40, zorder=5)
        ax.scatter(r.depth + 0.1, ratio_4x, c="tab:purple", marker="^", s=40, zorder=5)
    ax.scatter([], [], c="tab:green", marker="o", label="In-dist ratio")
    ax.scatter([], [], c="tab:purple", marker="^", label="4× extrap ratio")
    ax.set_yscale("log")
    ax.set_xlabel("Chain depth")
    ax.set_ylabel("Neural MSE / Compiled MSE")
    ax.set_title("Error Amplification Factor")
    ax.legend()
    ax.set_xticks(depths)

    # Panel 6: Example chain output comparison
    ax = axes[1, 2]
    example_chain = CHAIN_SPECS[4]  # 4-stage: sin -> square -> add_one -> sqrt_abs
    chain_name, chain_mods = example_chain
    x_plot = torch.linspace(-8, 8, 1000)
    gt = ground_truth_chain(chain_mods, x_plot)
    ax.plot(x_plot.numpy(), gt.numpy(), "k-", linewidth=2, label="Ground truth")

    x_train_boundary = 2.0
    ax.axvline(-x_train_boundary, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x_train_boundary, color="gray", linestyle="--", alpha=0.5)
    ax.text(0, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else -0.5,
            "train range", ha="center", va="bottom", fontsize=8, color="gray")

    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title(f"Example: {chain_name}")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compositional generalization experiment")
    parser.add_argument("--train-range", type=float, default=2.0)
    parser.add_argument("--neural-epochs", type=int, default=5000)
    parser.add_argument("--neural-hidden", type=int, default=32)
    parser.add_argument("--neural-layers", type=int, default=2)
    parser.add_argument("--n-eval", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-fig", default="examples/compositional_generalization.png")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip training/evaluation; load saved data and regenerate figures")
    args = parser.parse_args()

    # Derive pickle path from the figure path
    data_path = args.save_fig.replace(".png", "_data.pkl")

    if args.plot_only:
        if not os.path.exists(data_path):
            print(f"Error: data file not found: {data_path}")
            print("Run the experiment first (without --plot-only) to generate data.")
            sys.exit(1)
        print(f"Loading saved data from {data_path} ...")
        with open(data_path, "rb") as f:
            saved = pickle.load(f)
        results = saved["results"]
        module_errors = saved["module_errors"]
        visualize(results, module_errors, args.save_fig)
        return

    torch.manual_seed(args.seed)

    print("Compositional Generalization Experiment")
    print(f"  Train range: [-{args.train_range}, {args.train_range}]")
    print(f"  Neural: {args.neural_hidden}h × {args.neural_layers}L, {args.neural_epochs} epochs")
    print()

    # --- Compile all modules ---
    print("Compiling modules as frozen compiled modules...")
    compiled_modules: dict[str, DirectModule] = {}
    for name, source in MODULE_SPECS:
        graph = compile_scheme(source, inputs={"x": None})
        sg = DirectModule(graph)
        n_nodes = len(graph.nodes)
        compiled_modules[name] = sg
        print(f"  {name}: {source} → {n_nodes} nodes")
    print()

    # --- Train neural approximations ---
    print("Training neural approximations...")
    neural_modules: dict[str, NeuralModule] = {}
    module_errors: dict[str, float] = {}
    for name, _ in MODULE_SPECS:
        model, final_mse = train_neural_module(
            name, hidden=args.neural_hidden, layers=args.neural_layers,
            epochs=args.neural_epochs, train_range=args.train_range, seed=args.seed
        )
        neural_modules[name] = model
        module_errors[name] = final_mse
        print(f"  {name}: final MSE = {final_mse:.2e}")
    print()

    # --- Verify compiled modules produce exact results ---
    print("Verifying compiled module exactness...")
    x_test = torch.linspace(-args.train_range, args.train_range, 1000)
    for name, _ in MODULE_SPECS:
        gt = ground_truth_module(name, x_test)
        with torch.no_grad():
            pred = compiled_modules[name].forward_batch({"x": x_test})
        max_err = (pred - gt).abs().max().item()
        print(f"  {name}: max error = {max_err:.2e}")
    print()

    # --- Evaluate all chains ---
    print("Evaluating composition chains...")
    print("=" * 80)
    results: list[ChainResult] = []
    for chain_name, chain_mods in CHAIN_SPECS:
        r = evaluate_chain(
            chain_name, chain_mods, compiled_modules, neural_modules,
            n_eval=args.n_eval, train_range=args.train_range
        )
        results.append(r)
        ratio = r.neural_mse / max(r.compiled_mse, 1e-16)
        print(f"  [{r.depth}-stage] {chain_name}")
        print(f"    Compiled:   MSE={r.compiled_mse:.2e}, max={r.compiled_max_err:.2e}")
        print(f"    Hand-coded: MSE={r.handcoded_mse:.2e}, max={r.handcoded_max_err:.2e}")
        print(f"    Neural:     MSE={r.neural_mse:.2e}, max={r.neural_max_err:.2e}")
        print(f"    Neural/Compiled ratio: {ratio:.1e}x")
        print()

    # --- Summary ---
    print("=" * 80)
    print("COMPOSITIONAL GENERALIZATION — RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Chain':<50} {'Depth':>5} {'Compiled MSE':>14} {'Neural MSE':>14} {'Ratio':>12}")
    print("-" * 95)
    for r in results:
        ratio = r.neural_mse / max(r.compiled_mse, 1e-16)
        print(f"{r.name:<50} {r.depth:>5} {r.compiled_mse:>14.2e} {r.neural_mse:>14.2e} {ratio:>12.1e}")

    print()
    print("EXTRAPOLATION (4×):")
    print(f"{'Chain':<50} {'Depth':>5} {'Compiled MSE':>14} {'Neural MSE':>14} {'Ratio':>12}")
    print("-" * 95)
    for r in results:
        ratio = r.neural_mse_4x / max(r.compiled_mse_4x, 1e-16)
        print(f"{r.name:<50} {r.depth:>5} {r.compiled_mse_4x:>14.2e} {r.neural_mse_4x:>14.2e} {ratio:>12.1e}")

    print()
    print("ERROR GROWTH BY DEPTH:")
    for depth in sorted(set(r.depth for r in results)):
        rs = [r for r in results if r.depth == depth]
        avg_c = sum(r.compiled_mse for r in rs) / len(rs)
        avg_n = sum(r.neural_mse for r in rs) / len(rs)
        avg_c4 = sum(r.compiled_mse_4x for r in rs) / len(rs)
        avg_n4 = sum(r.neural_mse_4x for r in rs) / len(rs)
        print(f"  Depth {depth}: compiled_avg={avg_c:.2e}, neural_avg={avg_n:.2e}, "
              f"extrap_compiled={avg_c4:.2e}, extrap_neural={avg_n4:.2e}")

    # --- Save data for later re-plotting ---
    with open(data_path, "wb") as f:
        pickle.dump({"results": results, "module_errors": module_errors}, f)
    print(f"\nPlot data saved to {data_path}")
    print("  (re-run with --plot-only to regenerate figures without re-training)")

    # --- Visualization ---
    visualize(results, module_errors, args.save_fig)


if __name__ == "__main__":
    main()

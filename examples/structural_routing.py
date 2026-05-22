############################################################
#
# Neural Compiler: Compiling scheme into composable and
#                  differentiable neural network representations
#
# structural_routing.py: Structural features from compiled modules improving router learning and sample efficiency
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Structural routing: compiled modules expose internal computation to routers.

Demonstrates that structural features (intermediate node values from
compiled subgraphs) provide a learning advantage for module identification:
faster convergence, better sample efficiency, and robustness under noise.

Twelve force laws are compiled. A router must identify which law generated
each observation. Three router variants are compared across multiple
conditions (noise levels, sample sizes, confusion regimes):
  A) Structural: input x, observed y, all module outputs, AND intermediate features
  B) Output-only: input x, observed y, and all module outputs
  C) Pure MLP: input x and observed y only (no compiled modules)

The structural features add no NEW information (they're determined by x), but
they provide a computational shortcut: the router doesn't have to learn to
reconstruct intermediate values from x. This translates to better sample
efficiency and learning speed.
"""

import argparse
import sys
import os
import time

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


FORCE_LAWS = [
    ("hooke",        "(- 0 x)"),
    ("pendulum",     "(- 0 (sin x))"),
    ("gravity",      "(- 0 (/ 1 (+ (* x x) 1)))"),
    ("coulomb",      "(/ 1 (+ (* x x) 1))"),
    ("stiff_spring", "(* (- 0 2) x)"),
    ("quad_drag",    "(* (- 0 1) (* x x))"),
    ("exp_decay",    "(* (- 0 1) (exp (- 0 x)))"),
    ("cubic_spring", "(* (- 0 1) (* x (* x x)))"),
    ("log_force",    "(* (- 0 1) (log (+ (* x x) 1)))"),
    ("cos_force",    "(cos x)"),
    ("duffing",      "(+ (- 0 x) (* (- 0 0.1) (* x (* x x))))"),
    ("sqrt_force",   "(* (- 0 1) (sqrt (+ (* x x) 0.01)))"),
]

N_LAWS = len(FORCE_LAWS)

CONFUSION_PAIRS = [
    ("hooke", "pendulum"),
    ("hooke", "duffing"),
    ("hooke", "cubic_spring"),
    ("gravity", "coulomb"),
]


def extract_structural_features(sg: DirectModule, x: torch.Tensor,
                                max_features: int = 20):
    batch_size = x.shape[0]
    device = x.device

    output, intermediates = sg.forward_batch_with_intermediates({"x": x})

    total = intermediates.shape[1]
    if total < max_features:
        pad = torch.zeros(batch_size, max_features - total, device=device)
        structural = torch.cat([intermediates, pad], dim=1)
    elif total > max_features:
        structural = intermediates[:, :max_features]
    else:
        structural = intermediates

    return output, structural


def ground_truth_law(name: str, x: torch.Tensor) -> torch.Tensor:
    if name == "hooke":        return -x
    elif name == "pendulum":   return -torch.sin(x)
    elif name == "gravity":    return -1.0 / (x * x + 1)
    elif name == "coulomb":    return 1.0 / (x * x + 1)
    elif name == "stiff_spring": return -2 * x
    elif name == "quad_drag":  return -x * x
    elif name == "exp_decay":  return -torch.exp(-x)
    elif name == "cubic_spring": return -x * x * x
    elif name == "log_force":  return -torch.log(x * x + 1)
    elif name == "cos_force":  return torch.cos(x)
    elif name == "duffing":    return -x - 0.1 * x * x * x
    elif name == "sqrt_force": return -torch.sqrt(x * x + 0.01)
    raise ValueError(name)


def generate_dataset(n_per_law, x_range, noise_level, seed):
    torch.manual_seed(seed)
    all_x, all_y, all_labels = [], [], []
    for idx, (name, _) in enumerate(FORCE_LAWS):
        x = torch.rand(n_per_law) * 2 * x_range - x_range
        y = ground_truth_law(name, x) + noise_level * torch.randn(n_per_law)
        all_x.append(x)
        all_y.append(y)
        all_labels.append(torch.full((n_per_law,), idx, dtype=torch.long))
    return torch.cat(all_x), torch.cat(all_y), torch.cat(all_labels)


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

class RouterA(nn.Module):
    def __init__(self, n_laws, max_features, hidden=128):
        super().__init__()
        input_dim = 1 + 1 + n_laws + n_laws * max_features
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_laws))

    def forward(self, x):
        return self.net(x)


class RouterB(nn.Module):
    def __init__(self, n_laws, hidden=128):
        super().__init__()
        input_dim = 1 + 1 + n_laws
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_laws))

    def forward(self, x):
        return self.net(x)


class RouterC(nn.Module):
    def __init__(self, n_laws, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_laws))

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------

def prepare_features_A(x, y, compiled_modules, max_features):
    outputs, structs = [], []
    for name, _ in FORCE_LAWS:
        out, feat = extract_structural_features(compiled_modules[name], x, max_features)
        outputs.append(out.unsqueeze(1))
        structs.append(feat)
    return torch.cat([x.unsqueeze(1), y.unsqueeze(1)] + outputs + structs, dim=1)


def prepare_features_B(x, y, compiled_modules):
    outputs = []
    for name, _ in FORCE_LAWS:
        out = compiled_modules[name].forward_batch({"x": x})
        outputs.append(out.unsqueeze(1))
    return torch.cat([x.unsqueeze(1), y.unsqueeze(1)] + outputs, dim=1)


def prepare_features_C(x, y):
    return torch.cat([x.unsqueeze(1), y.unsqueeze(1)], dim=1)


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def train_router(model, features, labels, epochs, lr, seed):
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    n = features.shape[0]
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        bs = min(512, n)
        for i in range(0, n, bs):
            batch_f = features[perm[i:i+bs]]
            batch_l = labels[perm[i:i+bs]]
            loss = F.cross_entropy(model(batch_f), batch_l)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_f.shape[0]
        scheduler.step()
    return epoch_loss / n


def evaluate_accuracy(model, features, labels):
    model.eval()
    with torch.no_grad():
        preds = model(features).argmax(dim=1)
    acc = (preds == labels).float().mean().item()
    model.train()
    return acc


def per_class_accuracy(model, features, labels, n_classes):
    model.eval()
    with torch.no_grad():
        preds = model(features).argmax(dim=1)
    accs = []
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() == 0:
            accs.append(0.0)
        else:
            accs.append((preds[mask] == c).float().mean().item())
    model.train()
    return accs


def confusion_matrix(model, features, labels, n_classes):
    model.eval()
    with torch.no_grad():
        preds = model(features).argmax(dim=1)
    mat = torch.zeros(n_classes, n_classes, dtype=torch.long)
    for t, p in zip(labels, preds):
        mat[t, p] += 1
    model.train()
    return mat


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------

def run_single_condition(compiled_modules, max_features, hidden, epochs, lr,
                         n_train, n_test, x_range, noise, seed):
    """Train and evaluate all three routers for one (noise, sample_size) condition."""
    x_train, y_train, l_train = generate_dataset(n_train, x_range, noise, seed)
    x_test, y_test, l_test = generate_dataset(n_test, x_range, noise, seed + 100)
    x_conf, y_conf, l_conf = generate_dataset(n_test, 0.3, noise, seed + 200)

    with torch.no_grad():
        fA_train = prepare_features_A(x_train, y_train, compiled_modules, max_features)
        fA_test = prepare_features_A(x_test, y_test, compiled_modules, max_features)
        fA_conf = prepare_features_A(x_conf, y_conf, compiled_modules, max_features)
        fB_train = prepare_features_B(x_train, y_train, compiled_modules)
        fB_test = prepare_features_B(x_test, y_test, compiled_modules)
        fB_conf = prepare_features_B(x_conf, y_conf, compiled_modules)
        fC_train = prepare_features_C(x_train, y_train)
        fC_test = prepare_features_C(x_test, y_test)
        fC_conf = prepare_features_C(x_conf, y_conf)

    results = {}
    for label, Router, f_tr, f_te, f_co in [
        ("A", RouterA, fA_train, fA_test, fA_conf),
        ("B", RouterB, fB_train, fB_test, fB_conf),
        ("C", RouterC, fC_train, fC_test, fC_conf),
    ]:
        if label == "A":
            model = Router(N_LAWS, max_features, hidden=hidden)
        else:
            model = Router(N_LAWS, hidden=hidden)
        train_router(model, f_tr, l_train, epochs, lr, seed)
        model.eval()
        results[label] = {
            "test": evaluate_accuracy(model, f_te, l_test),
            "confusion": evaluate_accuracy(model, f_co, l_conf),
            "per_class": per_class_accuracy(model, f_te, l_test, N_LAWS),
            "conf_mat": confusion_matrix(model, f_te, l_test, N_LAWS),
        }
    return results


def main():
    parser = argparse.ArgumentParser(description="Structural routing experiment")
    parser.add_argument("--max-features", type=int, default=20)
    parser.add_argument("--router-hidden", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-test", type=int, default=200)
    parser.add_argument("--save-fig", default="examples/structural_routing.png")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    print("Structural Routing Experiment")
    print(f"  Force laws: {N_LAWS}")
    print(f"  Structural features: {args.max_features} dims per module")
    print()

    # --- Compile ---
    print("Compiling force laws...")
    compiled_modules: dict[str, DirectModule] = {}
    for name, source in FORCE_LAWS:
        graph = compile_scheme(source, inputs={"x": None})
        sg = DirectModule(graph)
        compiled_modules[name] = sg
        print(f"  {name}: {source} → {len(graph.nodes)} nodes")
    print()

    # --- Verify ---
    print("Verifying compiled modules...")
    x_v = torch.linspace(-2, 2, 1000)
    for name, _ in FORCE_LAWS:
        gt = ground_truth_law(name, x_v)
        with torch.no_grad():
            pred = compiled_modules[name].forward_batch({"x": x_v})
        print(f"  {name}: max error = {(pred - gt).abs().max().item():.2e}")
    print()

    # ================================================================
    # ANALYSIS 1: Sample efficiency sweep
    # ================================================================
    print("=" * 70)
    print("ANALYSIS 1: SAMPLE EFFICIENCY")
    print("  Noise: 5%, x range: [-2, 2], varying training samples/law")
    print("=" * 70)

    sample_sizes = [50, 100, 200, 500, 1000]
    sample_results = {}
    for n_train in sample_sizes:
        print(f"\n  n_train={n_train}/law ({n_train * N_LAWS} total)...")
        r = run_single_condition(
            compiled_modules, args.max_features, args.router_hidden,
            args.epochs, args.lr, n_train, args.n_test, 2.0, 0.05, args.seed)
        sample_results[n_train] = r
        print(f"    A={r['A']['test']:.1%}  B={r['B']['test']:.1%}  C={r['C']['test']:.1%}")

    # ================================================================
    # ANALYSIS 2: Noise robustness sweep
    # ================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 2: NOISE ROBUSTNESS")
    print("  n_train=500/law, x range: [-2, 2], varying noise")
    print("=" * 70)

    noise_levels = [0.01, 0.05, 0.10, 0.20]
    noise_results = {}
    for noise in noise_levels:
        print(f"\n  noise={noise:.0%}...")
        r = run_single_condition(
            compiled_modules, args.max_features, args.router_hidden,
            args.epochs, args.lr, 500, args.n_test, 2.0, noise, args.seed)
        noise_results[noise] = r
        print(f"    A={r['A']['test']:.1%}  B={r['B']['test']:.1%}  C={r['C']['test']:.1%}")
        print(f"    Confusion: A={r['A']['confusion']:.1%}  B={r['B']['confusion']:.1%}  C={r['C']['confusion']:.1%}")

    # ================================================================
    # ANALYSIS 3: Confusion regime (small |x|, high noise)
    # ================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 3: CONFUSION REGIME")
    print("  n_train=1000/law, x range: [-0.5, 0.5], noise: 5%")
    print("=" * 70)

    r_conf = run_single_condition(
        compiled_modules, args.max_features, args.router_hidden,
        args.epochs, args.lr, 1000, args.n_test, 0.5, 0.05, args.seed)
    print(f"\n  Overall: A={r_conf['A']['test']:.1%}  B={r_conf['B']['test']:.1%}  C={r_conf['C']['test']:.1%}")
    print("\n  Per-law accuracy:")
    law_names = [n for n, _ in FORCE_LAWS]
    for i, name in enumerate(law_names):
        a = r_conf["A"]["per_class"][i]
        b = r_conf["B"]["per_class"][i]
        c = r_conf["C"]["per_class"][i]
        marker = " ← confusion" if name in {"hooke", "pendulum", "duffing"} else ""
        print(f"    {name:<15} A={a:.1%}  B={b:.1%}  C={c:.1%}{marker}")

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 80)
    print("STRUCTURAL ROUTING — RESULTS SUMMARY")
    print("=" * 80)

    print("\nSample Efficiency (5% noise, full range):")
    print(f"  {'n/law':>8}  {'A (struct)':>12}  {'B (output)':>12}  {'C (MLP)':>12}  {'A-B':>8}  {'A-C':>8}")
    print("  " + "-" * 65)
    for n in sample_sizes:
        r = sample_results[n]
        a, b, c = r["A"]["test"], r["B"]["test"], r["C"]["test"]
        print(f"  {n:>8}  {a:>12.1%}  {b:>12.1%}  {c:>12.1%}  {a-b:>+8.1%}  {a-c:>+8.1%}")

    print("\nNoise Robustness (500/law, full range):")
    print(f"  {'noise':>8}  {'A (struct)':>12}  {'B (output)':>12}  {'C (MLP)':>12}  {'A-B':>8}  {'A-C':>8}")
    print("  " + "-" * 65)
    for noise in noise_levels:
        r = noise_results[noise]
        a, b, c = r["A"]["test"], r["B"]["test"], r["C"]["test"]
        print(f"  {noise:>8.0%}  {a:>12.1%}  {b:>12.1%}  {c:>12.1%}  {a-b:>+8.1%}  {a-c:>+8.1%}")

    print("\nNoise Robustness — Confusion Regime (|x| < 0.3):")
    print(f"  {'noise':>8}  {'A (struct)':>12}  {'B (output)':>12}  {'C (MLP)':>12}  {'A-B':>8}")
    print("  " + "-" * 55)
    for noise in noise_levels:
        r = noise_results[noise]
        a, b, c = r["A"]["confusion"], r["B"]["confusion"], r["C"]["confusion"]
        print(f"  {noise:>8.0%}  {a:>12.1%}  {b:>12.1%}  {c:>12.1%}  {a-b:>+8.1%}")

    print(f"\nConfusion Regime Only (|x| < 0.5, 5% noise):")
    a, b, c = r_conf["A"]["test"], r_conf["B"]["test"], r_conf["C"]["test"]
    print(f"  A={a:.1%}  B={b:.1%}  C={c:.1%}  (A-B={a-b:+.1%}, A-C={a-c:+.1%})")

    # --- Confusion pair analysis ---
    print("\nConfusion Pair Analysis (confusion regime test):")
    law_idx = {name: i for i, (name, _) in enumerate(FORCE_LAWS)}
    for law1, law2 in CONFUSION_PAIRS:
        i, j = law_idx[law1], law_idx[law2]
        print(f"  {law1} ↔ {law2}:")
        for label in ["A", "B", "C"]:
            mat = r_conf[label]["conf_mat"].float()
            mat_norm = mat / mat.sum(dim=1, keepdim=True).clamp(min=1)
            cross = mat_norm[i, j].item() + mat_norm[j, i].item()
            print(f"    Router {label}: cross-confusion rate = {cross/2:.1%}")

    # --- Visualization ---
    visualize(sample_results, noise_results, r_conf, sample_sizes, noise_levels,
              law_names, args.save_fig)


def visualize(sample_results, noise_results, conf_results,
              sample_sizes, noise_levels, law_names, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Experiment 3B: Structural Routing with Compiled Compiled Modules",
                 fontsize=14, fontweight="bold")
    colors = {"A": "tab:blue", "B": "tab:orange", "C": "tab:red"}
    labels_map = {"A": "Structural (A)", "B": "Output-only (B)", "C": "Pure MLP (C)"}

    # Panel 1: Sample efficiency (test accuracy)
    ax = axes[0, 0]
    for key in ["A", "B", "C"]:
        accs = [sample_results[n][key]["test"] for n in sample_sizes]
        ax.plot(sample_sizes, accs, "o-", color=colors[key], label=labels_map[key])
    ax.set_xlabel("Training samples per law")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Sample Efficiency (5% noise)")
    ax.legend(fontsize=8)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)

    # Panel 2: Noise robustness (test accuracy)
    ax = axes[0, 1]
    noise_pct = [n * 100 for n in noise_levels]
    for key in ["A", "B", "C"]:
        accs = [noise_results[n][key]["test"] for n in noise_levels]
        ax.plot(noise_pct, accs, "o-", color=colors[key], label=labels_map[key])
    ax.set_xlabel("Noise level (%)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Noise Robustness (500/law)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)

    # Panel 3: Noise robustness — confusion regime
    ax = axes[0, 2]
    for key in ["A", "B", "C"]:
        accs = [noise_results[n][key]["confusion"] for n in noise_levels]
        ax.plot(noise_pct, accs, "o-", color=colors[key], label=labels_map[key])
    ax.set_xlabel("Noise level (%)")
    ax.set_ylabel("Accuracy (|x| < 0.3)")
    ax.set_title("Confusion Regime vs Noise")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)

    # Panel 4: Per-law accuracy (confusion regime experiment)
    ax = axes[1, 0]
    x_pos = np.arange(N_LAWS)
    w = 0.25
    for offset, key in [(-w, "A"), (0, "B"), (w, "C")]:
        accs = conf_results[key]["per_class"]
        ax.bar(x_pos + offset, accs, w, label=labels_map[key],
               color=colors[key], alpha=0.8)
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-Law: Confusion Regime (|x|<0.5)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(law_names, rotation=45, ha="right", fontsize=6)
    ax.legend(fontsize=7)
    ax.set_ylim(0, 1.15)

    # Panel 5: Confusion matrix — Router B (output-only)
    ax = axes[1, 1]
    mat = conf_results["B"]["conf_mat"].float()
    mat = mat / mat.sum(dim=1, keepdim=True).clamp(min=1)
    ax.imshow(mat.numpy(), cmap="Reds", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(N_LAWS))
    ax.set_yticks(range(N_LAWS))
    ax.set_xticklabels(law_names, rotation=45, ha="right", fontsize=5)
    ax.set_yticklabels(law_names, fontsize=5)
    ax.set_title("Confusion: Output-Only (B)")

    # Panel 6: Force laws in confusion zone
    ax = axes[1, 2]
    x_plot = torch.linspace(-0.5, 0.5, 500)
    highlight = {"hooke", "pendulum", "duffing", "cubic_spring"}
    for name, _ in FORCE_LAWS:
        y_plot = ground_truth_law(name, x_plot).numpy()
        if name in highlight:
            ax.plot(x_plot.numpy(), y_plot, linewidth=2, label=name)
        else:
            ax.plot(x_plot.numpy(), y_plot, linewidth=0.5, alpha=0.3, color="gray")
    ax.set_xlabel("x")
    ax.set_ylabel("F(x)")
    ax.set_title("Force Laws in Confusion Zone")
    ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    main()

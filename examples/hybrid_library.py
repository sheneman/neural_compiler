#!/usr/bin/env python3
############################################################
#
# Neural Compiler: Compiling scheme into composable and
#                  differentiable neural network representations
#
# hybrid_library.py: Trainable model discovering which compiled subgraphs to select from a library
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""
Hybrid Example: Learned Subgraph Selection from a Library

A trainable model discovers which compiled subgraphs to use from a library
of 16 programs, learns input projections for each, and combines selected
outputs. L1 regularization on output weights drives sparse selection.

Target: f(x,y) = (x+y)³ + 2((x-y)² + 1) + 3(x+y) - 1
      = (x+y)³ + 2(x-y)² + 3(x+y) + 1

Canonical decomposition (3 of 16):
  subgraph[1]  x³   : proj=[1,1]  → (x+y)³,     weight=1.0
  subgraph[2]  x²+1 : proj=[1,-1] → (x-y)²+1,   weight=2.0
  subgraph[5]  2x   : proj=[1,1]  → 2(x+y),     weight=1.5
  bias = -1.0
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_compiler.compiler import compile_scheme
from neural_compiler.evaluator import DirectModule

LIBRARY = [
    ("x²",     "(* x x)"),
    ("x³",     "(let ((x2 (* x x))) (* x x2))"),
    ("x²+1",   "(+ (* x x) 1)"),
    ("x²-1",   "(- (* x x) 1)"),
    ("x⁴",     "(let ((x2 (* x x))) (* x2 x2))"),
    ("2x",     "(* 2 x)"),
    ("x+1",    "(+ x 1)"),
    ("x-1",    "(- x 1)"),
    ("x²+x",   "(+ (* x x) x)"),
    ("x²-x",   "(- (* x x) x)"),
    ("x⁴+x²",  "(let ((x2 (* x x))) (+ (* x2 x2) x2))"),
    ("x⁴-x²",  "(let ((x2 (* x x))) (- (* x2 x2) x2))"),
    ("3x²+2x", "(+ (* 3 (* x x)) (* 2 x))"),
    ("x²-2x",  "(- (* x x) (* 2 x))"),
    ("x⁶",     "(let ((x2 (* x x))) (* x2 (* x2 x2)))"),
    ("x³+x²",  "(let ((x2 (* x x))) (+ (* x x2) x2))"),
]

TRUE_ACTIVE = {1, 2, 5}


def target_fn(x, y):
    s = x + y
    d = x - y
    return s**3 + 2 * (d**2 + 1) + 3 * s - 1


class HybridLibraryModel(nn.Module):
    def __init__(self, subgraphs):
        super().__init__()
        self.n_lib = len(subgraphs)
        self.subgraphs = nn.ModuleList(subgraphs)
        self.projections = nn.ModuleList(
            [nn.Linear(2, 1, bias=False) for _ in range(self.n_lib)]
        )
        self.output_weights = nn.Linear(self.n_lib, 1)

    def forward(self, x, y):
        inp = torch.stack([x, y], dim=1)
        outputs = []
        for proj, sg in zip(self.projections, self.subgraphs):
            z = proj(inp).squeeze(1)
            outputs.append(sg.forward_batch({"x": z}))
        features = torch.stack(outputs, dim=1)
        return self.output_weights(features).squeeze(1)

    def l1_penalty(self):
        return self.output_weights.weight.abs().sum()

    def active_indices(self, threshold=0.01):
        w = self.output_weights.weight.detach().squeeze()
        return {i for i in range(self.n_lib) if w[i].abs().item() > threshold}


class PureMLPModel(nn.Module):
    def __init__(self, hidden=64, layers=4):
        super().__init__()
        mods = [nn.Linear(2, hidden), nn.ReLU()]
        for _ in range(layers - 1):
            mods.extend([nn.Linear(hidden, hidden), nn.ReLU()])
        mods.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*mods)

    def forward(self, x, y):
        return self.net(torch.stack([x, y], dim=1)).squeeze(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--l1-weight", type=float, default=0.5)
    parser.add_argument("--l1-warmup", type=float, default=0.2,
                        help="Fraction of epochs to ramp L1 from 0 to l1_weight")
    parser.add_argument("--train-range", type=float, default=2.0)
    parser.add_argument("--extrap-range", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Compiling library of {len(LIBRARY)} subgraphs...")
    subgraphs = []
    total_nodes, total_edges, n_consts = 0, 0, 0
    for name, source in LIBRARY:
        graph = compile_scheme(source, inputs={"x": None})
        sg = DirectModule(graph)
        subgraphs.append(sg)
        n_n = len(graph.nodes)
        n_e = sum(len(nd.input_edges) for nd in graph.nodes.values())
        n_c = sum(1 for nd in graph.nodes.values() if nd.op_type == "const")
        total_nodes += n_n
        total_edges += n_e
        n_consts += n_c
        print(f"  [{name:>8s}] {source:45s} → {n_n} nodes")

    print(f"\nTotal frozen structure: {total_nodes} nodes, {total_edges} edges, {n_consts} consts")

    hybrid = HybridLibraryModel(subgraphs)
    mlp = PureMLPModel(hidden=64, layers=4)

    h_params = sum(p.numel() for p in hybrid.parameters() if p.requires_grad)
    m_params = sum(p.numel() for p in mlp.parameters())
    proj_params = sum(p.numel() for p in hybrid.projections.parameters())
    out_params = sum(p.numel() for p in hybrid.output_weights.parameters())

    print(f"\nHybrid trainable params: {h_params}")
    print(f"  Projections: {proj_params} ({len(LIBRARY)} × Linear(2→1))")
    print(f"  Output combination: {out_params} (Linear({len(LIBRARY)}→1))")
    print(f"  Frozen: {total_nodes} nodes, {total_edges} edges, {n_consts} consts")
    print(f"MLP baseline trainable params: {m_params}")

    opt_h = torch.optim.Adam(hybrid.parameters(), lr=args.lr)
    opt_m = torch.optim.Adam(mlp.parameters(), lr=args.lr)

    R = args.train_range
    hist_h, hist_m = [], []

    warmup_epochs = int(args.epochs * args.l1_warmup)
    print(f"\nTraining ({args.epochs} epochs, L1 max={args.l1_weight}, warmup={warmup_epochs} epochs)...")
    for epoch in range(args.epochs):
        l1_scale = min(1.0, epoch / warmup_epochs) if warmup_epochs > 0 else 1.0
        l1_w = args.l1_weight * l1_scale

        x = torch.empty(args.batch_size).uniform_(-R, R)
        y = torch.empty(args.batch_size).uniform_(-R, R)
        t = target_fn(x, y)

        opt_h.zero_grad()
        pred_h = hybrid(x, y)
        mse_h = nn.functional.mse_loss(pred_h, t)
        loss_h = mse_h + l1_w * hybrid.l1_penalty()
        loss_h.backward()
        opt_h.step()
        hist_h.append(mse_h.item())

        opt_m.zero_grad()
        pred_m = mlp(x, y)
        loss_m = nn.functional.mse_loss(pred_m, t)
        loss_m.backward()
        opt_m.step()
        hist_m.append(loss_m.item())

        if epoch % 500 == 0 or epoch == args.epochs - 1:
            active = hybrid.active_indices()
            print(
                f"  Epoch {epoch:5d}: hybrid MSE={mse_h.item():.6f} "
                f"(L1={hybrid.l1_penalty().item():.4f}, λ={l1_w:.3f}, "
                f"active={sorted(active)})  MLP MSE={loss_m.item():.6f}"
            )

    # --- Results ---
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    weights = hybrid.output_weights.weight.detach().squeeze()
    bias = hybrid.output_weights.bias.detach().item()

    print("\nLibrary selection (output weights):")
    print(f"  {'Idx':>3s}  {'Name':>8s}  {'Weight':>10s}  {'Status'}")
    print(f"  {'---':>3s}  {'----':>8s}  {'------':>10s}  {'------'}")
    for i, (name, _) in enumerate(LIBRARY):
        w = weights[i].item()
        status = ""
        if abs(w) > 0.01:
            status += "ACTIVE"
        if i in TRUE_ACTIVE:
            status += " (true)" if status else "(true, pruned!)"
        print(f"  [{i:2d}]  {name:>8s}  {w:+10.4f}  {status}")
    print(f"  {'':>3s}  {'bias':>8s}  {bias:+10.4f}")

    active = hybrid.active_indices()
    print(f"\nDiscovered active set: {sorted(active)}")
    print(f"True active set:      {sorted(TRUE_ACTIVE)}")
    correct = active == TRUE_ACTIVE
    superset = TRUE_ACTIVE.issubset(active)
    print(f"Exact match: {correct}")
    if not correct and superset:
        extra = active - TRUE_ACTIVE
        print(f"All true subgraphs found, {len(extra)} extra: {sorted(extra)}")

    print("\nLearned projections (active subgraphs):")
    true_projs = {1: "[1, 1]", 2: "[1, -1]", 5: "[1, 1]"}
    true_weights = {1: 1.0, 2: 2.0, 5: 1.5}
    for i in sorted(active):
        p = hybrid.projections[i].weight.detach().squeeze().tolist()
        w = weights[i].item()
        name = LIBRARY[i][0]
        tp = true_projs.get(i, "—")
        tw = true_weights.get(i, "—")
        print(f"  [{i}] {name:>8s}: proj=[{p[0]:+.4f}, {p[1]:+.4f}]  w={w:+.4f}  (true proj={tp}, w={tw})")

    # Equivalent factorization analysis
    print("\nEquivalent factorization analysis:")
    for i in sorted(active & TRUE_ACTIVE):
        p = hybrid.projections[i].weight.detach().squeeze().tolist()
        w = weights[i].item()
        if i == 1:  # x³: proj≈α[1,1], so cube(α(x+y))=α³(x+y)³, w·α³ should=1
            alpha = (p[0] ** 2 + p[1] ** 2) ** 0.5
            direction = f"[{p[0] / alpha:+.2f}, {p[1] / alpha:+.2f}]"
            print(f"  x³: direction={direction}, scale α={alpha:.4f}")
            print(f"       w·α³ = {w:.4f}·{p[0]:.4f}³ = {w * p[0] ** 3:.4f} (should ≈ 1.0)")
        elif i == 2:  # x²+1: proj≈β[-1,1], input=β(y-x)=-β(x-y), squared: β²(x-y)²
            beta = p[0]  # proj ≈ [β, -β] → input = β·x + (-β)·y = β(x-y)
            print(f"  x²+1: proj=[{p[0]:+.2f}, {p[1]:+.2f}] ≈ {beta:.2f}·[1,-1], w={w:.4f}")
            print(f"         w·β² = {w:.4f}·{beta:.2f}² = {w * beta ** 2:.4f} (should ≈ 2.0)")
            print(f"         w + bias = {w:.4f} + {bias:.4f} = {w + bias:.4f} (should ≈ 1.0)")
            print(f"         L1 minimizes w by inflating β; constant absorbed by bias")
        elif i == 5:  # 2x: proj≈γ[1,1], so 2γ(x+y), w·2γ should=3
            print(f"  2x: proj=[{p[0]:+.4f}, {p[1]:+.4f}], w={w:.4f}")
            print(f"      w·2·proj[0] = {w * 2 * p[0]:.4f} (should ≈ 3.0)")

    with torch.no_grad():
        x_t = torch.empty(10000).uniform_(-R, R)
        y_t = torch.empty(10000).uniform_(-R, R)
        t_t = target_fn(x_t, y_t)
        mse_h_test = nn.functional.mse_loss(hybrid(x_t, y_t), t_t).item()
        mse_m_test = nn.functional.mse_loss(mlp(x_t, y_t), t_t).item()

    print(f"\nTest MSE (in-distribution [-{R}, {R}]):")
    print(f"  Hybrid:   {mse_h_test:.6f}")
    print(f"  Baseline: {mse_m_test:.6f}")
    if mse_h_test > 0 and mse_m_test > 0:
        if mse_h_test < mse_m_test:
            print(f"  Ratio:    Hybrid is {mse_m_test / mse_h_test:.0f}x better")
        else:
            print(f"  Ratio:    Baseline is {mse_h_test / mse_m_test:.1f}x better")

    E = args.extrap_range
    with torch.no_grad():
        x_e = torch.empty(10000).uniform_(R, E)
        y_e = torch.empty(10000).uniform_(R, E)
        t_e = target_fn(x_e, y_e)
        mse_h_ext = nn.functional.mse_loss(hybrid(x_e, y_e), t_e).item()
        mse_m_ext = nn.functional.mse_loss(mlp(x_e, y_e), t_e).item()

    print(f"\nTest MSE (extrapolation [{R}, {E}]):")
    print(f"  Hybrid:   {mse_h_ext:.6f}")
    print(f"  Baseline: {mse_m_ext:.6f}")
    if mse_h_ext > 0 and mse_m_ext > 0:
        if mse_h_ext < mse_m_ext:
            print(f"  Ratio:    Hybrid is {mse_m_ext / mse_h_ext:.0f}x better")
        else:
            print(f"  Ratio:    Baseline is {mse_h_ext / mse_m_ext:.1f}x better")

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Library Selection: {len(active)} Active from {len(LIBRARY)} Compiled Subgraphs",
        fontsize=14,
    )

    # Training loss
    ax = axes[0, 0]
    ax.semilogy(hist_h, alpha=0.7, label=f"Hybrid ({h_params} params)")
    ax.semilogy(hist_m, alpha=0.7, label=f"MLP ({m_params} params)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (log scale)")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Library weights
    ax = axes[0, 1]
    names = [n for n, _ in LIBRARY]
    ws = weights.numpy()
    colors = ["#2ecc71" if i in active & TRUE_ACTIVE else
              "#e74c3c" if i in active - TRUE_ACTIVE else
              "#f39c12" if i in TRUE_ACTIVE - active else
              "#bdc3c7" for i in range(len(LIBRARY))]
    ax.bar(range(len(LIBRARY)), ws, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(LIBRARY)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Output Weight")
    ax.set_title("Library Selection Weights (L1-regularized)")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.axhline(y=0.01, color="red", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.axhline(y=-0.01, color="red", linewidth=0.5, linestyle="--", alpha=0.5)
    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(color="#2ecc71", label="Correctly selected"),
            Patch(color="#bdc3c7", label="Correctly pruned"),
            Patch(color="#e74c3c", label="False positive"),
            Patch(color="#f39c12", label="Missed (false negative)"),
        ],
        fontsize=7,
        loc="upper right",
    )
    ax.grid(True, alpha=0.3, axis="y")

    # f(x, 0.5) slice
    ax = axes[1, 0]
    xp = torch.linspace(-E, E, 500)
    yf = torch.full_like(xp, 0.5)
    with torch.no_grad():
        tp = target_fn(xp, yf)
        hp = hybrid(xp, yf)
        mp = mlp(xp, yf)
    ax.plot(xp, tp, "k-", linewidth=2, label="True f(x, 0.5)")
    ax.plot(xp, hp, "g--", linewidth=1.5, label="Hybrid")
    ax.plot(xp, mp, "r:", linewidth=1.5, label="MLP")
    ax.axvspan(-R, R, alpha=0.1, color="blue", label="Training range")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x, 0.5)")
    ax.set_title("Extrapolation: f(x, 0.5)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # f(0.5, y) slice
    ax = axes[1, 1]
    yp = torch.linspace(-E, E, 500)
    xf = torch.full_like(yp, 0.5)
    with torch.no_grad():
        tp2 = target_fn(xf, yp)
        hp2 = hybrid(xf, yp)
        mp2 = mlp(xf, yp)
    ax.plot(yp, tp2, "k-", linewidth=2, label="True f(0.5, y)")
    ax.plot(yp, hp2, "g--", linewidth=1.5, label="Hybrid")
    ax.plot(yp, mp2, "r:", linewidth=1.5, label="MLP")
    ax.axvspan(-R, R, alpha=0.1, color="blue", label="Training range")
    ax.set_xlabel("y")
    ax.set_ylabel("f(0.5, y)")
    ax.set_title("Extrapolation: f(0.5, y)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hybrid_library.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to {fig_path}")


if __name__ == "__main__":
    main()

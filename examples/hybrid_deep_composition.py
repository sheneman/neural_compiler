#!/usr/bin/env python3
############################################################
#
# Neural Compiler: Compiling scheme into composable and
#                  differentiable neural network representations
#
# hybrid_deep_composition.py: Gradient flow through three frozen compiled subgraphs in series
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""
Hybrid Example: Deep Multi-Stage Composition (3 Frozen Layers)

Demonstrates gradient flow through THREE frozen compiled subgraphs in series.
A single 3-stage pipeline computes a degree-6 polynomial; the model learns
only an input projection and output scaling. Gradients must traverse all 3
frozen layers to train the projection weights.

Target: f(x,y) = ((x+y)^2 + 1)^3 + 2*(x-y) - 1

Pipeline: square -> add_one -> cube = ((proj(x,y))^2 + 1)^3
Direct:   2*(x-y) - 1 (linear terms, no pipeline)

Canonical: proj=[1,1], w_pipe=1.0, w_x=2, w_y=-2, bias=-1
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

PIPELINE_SPECS = [
    ("square",  "(* x x)"),
    ("add_one", "(+ x 1)"),
    ("cube",    "(let ((x2 (* x x))) (* x x2))"),
]


def target_fn(x, y):
    s = x + y
    return (s**2 + 1) ** 3 + 2 * (x - y) - 1


def compile_pipeline(specs, label):
    subgraphs = []
    total_nodes, total_edges, n_consts = 0, 0, 0
    parts = []
    for name, source in specs:
        graph = compile_scheme(source, inputs={"x": None})
        sg = DirectModule(graph)
        subgraphs.append(sg)
        n_n = len(graph.nodes)
        n_e = sum(len(nd.input_edges) for nd in graph.nodes.values())
        n_c = sum(1 for nd in graph.nodes.values() if nd.op_type == "const")
        total_nodes += n_n
        total_edges += n_e
        n_consts += n_c
        parts.append(f"{name}({n_n})")
    print(f"  {label}: {' → '.join(parts)} = {total_nodes} nodes, {total_edges} edges, {n_consts} consts")
    return subgraphs, total_nodes, total_edges, n_consts


class HybridDeepCompositionModel(nn.Module):
    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = nn.ModuleList(pipeline)
        self.proj = nn.Linear(2, 1, bias=False)
        self.output = nn.Linear(3, 1)

    def forward(self, x, y):
        inp = torch.stack([x, y], dim=1)

        z = self.proj(inp).squeeze(1)
        for sg in self.pipeline:
            z = sg.forward_batch({"x": z})

        features = torch.stack([z, x, y], dim=1)
        return self.output(features).squeeze(1)


class PureMLPModel(nn.Module):
    def __init__(self, hidden=64, layers=3):
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
    parser.add_argument("--epochs", type=int, default=15000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--train-range", type=float, default=1.0)
    parser.add_argument("--extrap-range", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("Compiling 3-stage pipeline (3 subgraphs)...")
    sgs, total_n, total_e, total_c = compile_pipeline(PIPELINE_SPECS, "Pipeline")

    hybrid = HybridDeepCompositionModel(sgs)
    mlp = PureMLPModel(hidden=64, layers=3)

    h_params = sum(p.numel() for p in hybrid.parameters() if p.requires_grad)
    m_params = sum(p.numel() for p in mlp.parameters())

    print(f"\nHybrid trainable params: {h_params}")
    print(f"  proj:   2 (Linear(2→1, no bias)) — gradient through 3 frozen subgraphs")
    print(f"  output: 4 (Linear(3→1) = 3 weights + 1 bias)")
    print(f"  Frozen: {total_n} nodes, {total_e} edges, {total_c} consts")
    print(f"MLP baseline trainable params: {m_params}")

    opt_h = torch.optim.Adam(hybrid.parameters(), lr=args.lr)
    opt_m = torch.optim.Adam(mlp.parameters(), lr=args.lr)

    R = args.train_range
    hist_h, hist_m = [], []

    print(f"\nTraining ({args.epochs} epochs, range [-{R}, {R}])...")
    for epoch in range(args.epochs):
        x = torch.empty(args.batch_size).uniform_(-R, R)
        y = torch.empty(args.batch_size).uniform_(-R, R)
        t = target_fn(x, y)

        opt_h.zero_grad()
        pred_h = hybrid(x, y)
        loss_h = nn.functional.mse_loss(pred_h, t)
        loss_h.backward()
        opt_h.step()
        hist_h.append(loss_h.item())

        opt_m.zero_grad()
        pred_m = mlp(x, y)
        loss_m = nn.functional.mse_loss(pred_m, t)
        loss_m.backward()
        opt_m.step()
        hist_m.append(loss_m.item())

        if epoch % 500 == 0 or epoch == args.epochs - 1:
            print(
                f"  Epoch {epoch:5d}: hybrid={loss_h.item():.6f}  "
                f"MLP={loss_m.item():.6f}"
            )

    # --- Gradient stability ---
    print("\nGradient stability check:")
    opt_h.zero_grad()
    x_chk = torch.empty(512).uniform_(-R, R)
    y_chk = torch.empty(512).uniform_(-R, R)
    t_chk = target_fn(x_chk, y_chk)
    loss_chk = nn.functional.mse_loss(hybrid(x_chk, y_chk), t_chk)
    loss_chk.backward()

    gp = hybrid.proj.weight.grad.norm().item()
    go = hybrid.output.weight.grad.norm().item()
    print(f"  proj   (3 frozen layers deep): ||grad|| = {gp:.6f}")
    print(f"  output (0 frozen layers):      ||grad|| = {go:.6f}")
    if go > 0:
        print(f"  Ratio deep/shallow:            {gp / go:.2f}x")

    # --- Results ---
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    proj = hybrid.proj.weight.detach().squeeze().tolist()
    out_w = hybrid.output.weight.detach().squeeze().tolist()
    out_b = hybrid.output.bias.detach().item()

    print("\nLearned parameters:")
    print(f"  proj:     [{proj[0]:+.4f}, {proj[1]:+.4f}]  (true: [1, 1] → x+y)")
    print(f"  output:   [{out_w[0]:+.4f}, {out_w[1]:+.4f}, {out_w[2]:+.4f}]  (true: [1.0, 2.0, -2.0])")
    print(f"  bias:     {out_b:+.4f}  (true: -1.0)")

    # Direction check
    norm_p = (proj[0] ** 2 + proj[1] ** 2) ** 0.5
    dir_p = [proj[0] / norm_p, proj[1] / norm_p] if norm_p > 0 else [0, 0]
    true_dir = [1 / 2**0.5, 1 / 2**0.5]
    cos_p = dir_p[0] * true_dir[0] + dir_p[1] * true_dir[1]
    print(f"\nProjection direction: [{dir_p[0]:+.4f}, {dir_p[1]:+.4f}], cos similarity = {abs(cos_p):.6f}")

    # Equivalent factorization
    alpha = (proj[0] + proj[1]) / 2
    print(f"\nEquivalent factorization:")
    print(f"  Effective projection: α·(x+y) where α = {alpha:.4f}")
    print(f"  Pipeline computes: (α²(x+y)² + 1)³")
    print(f"  α² = {alpha**2:.4f} (should = 1.0 for canonical)")
    val_origin = out_w[0] * (alpha**2 * 0 + 1) ** 3 + out_b
    val_true_origin = target_fn(torch.tensor(0.0), torch.tensor(0.0)).item()
    print(f"  w_pipe·1³ + bias = {val_origin:.4f} (should ≈ f(0,0)={val_true_origin:.4f})")
    print(f"  Linear weights: w_x={out_w[1]:+.4f} (true 2.0), w_y={out_w[2]:+.4f} (true -2.0)")

    # Test accuracy
    with torch.no_grad():
        x_t = torch.empty(10000).uniform_(-R, R)
        y_t = torch.empty(10000).uniform_(-R, R)
        t_t = target_fn(x_t, y_t)
        mse_h = nn.functional.mse_loss(hybrid(x_t, y_t), t_t).item()
        mse_m = nn.functional.mse_loss(mlp(x_t, y_t), t_t).item()

    print(f"\nTest MSE (in-distribution [-{R}, {R}]):")
    print(f"  Hybrid:   {mse_h:.6f}")
    print(f"  Baseline: {mse_m:.6f}")
    if mse_h > 0 and mse_m > 0:
        if mse_h < mse_m:
            print(f"  Ratio:    Hybrid is {mse_m / mse_h:.0f}x better")
        else:
            print(f"  Ratio:    Baseline is {mse_h / mse_m:.1f}x better")

    E = args.extrap_range
    with torch.no_grad():
        x_e = torch.empty(10000).uniform_(R, E)
        y_e = torch.empty(10000).uniform_(R, E)
        t_e = target_fn(x_e, y_e)
        mse_h_e = nn.functional.mse_loss(hybrid(x_e, y_e), t_e).item()
        mse_m_e = nn.functional.mse_loss(mlp(x_e, y_e), t_e).item()

    print(f"\nTest MSE (extrapolation [{R}, {E}]):")
    print(f"  Hybrid:   {mse_h_e:.6f}")
    print(f"  Baseline: {mse_m_e:.6f}")
    if mse_h_e > 0 and mse_m_e > 0:
        if mse_h_e < mse_m_e:
            print(f"  Ratio:    Hybrid is {mse_m_e / mse_h_e:.0f}x better")
        else:
            print(f"  Ratio:    Baseline is {mse_h_e / mse_m_e:.1f}x better")

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Deep Composition: Gradient Through 3 Frozen Subgraph Layers",
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

    # Learned weights
    ax = axes[0, 1]
    labels = ["proj[0]", "proj[1]", "w_pipe", "w_x", "w_y", "bias"]
    vals = [proj[0], proj[1], out_w[0], out_w[1], out_w[2], out_b]
    true_vals = [1, 1, 1.0, 2.0, -2.0, -1.0]
    x_pos = np.arange(len(labels))
    width = 0.35
    ax.bar(x_pos - width / 2, vals, width, label="Learned", color="#2ecc71", edgecolor="black", linewidth=0.5)
    ax.bar(x_pos + width / 2, true_vals, width, label="True (canonical)", color="#3498db", alpha=0.5, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Weight value")
    ax.set_title("Learned vs True Weights")
    ax.legend(fontsize=8)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    # f(x, 0.5)
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

    # f(0.5, y)
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
    fig_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "hybrid_deep_composition.png"
    )
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to {fig_path}")


if __name__ == "__main__":
    main()

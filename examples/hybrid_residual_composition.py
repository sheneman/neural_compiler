#!/usr/bin/env python3
############################################################
#
# Neural Compiler: Compiling scheme into composable and
#                  differentiable neural network representations
#
# hybrid_residual_composition.py: Residual connections and normalization at subgraph interfaces for deep composition
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""
Hybrid Example: Residual Deep Composition (Experiment 8)

Tests whether residual connections + normalization at subgraph interfaces
fix the optimization failure in 3-stage two-pipeline deep composition.

Previous result (Experiment 7): Two-pipeline architecture failed -- Pipeline B
(square->sub_one->square) trapped projection magnitude at beta~0.3. The sub_one
creates a gradient attractor via d/dbeta ~ -4*beta*s^2 at small beta.

Ablation with three hybrid variants:
  1. Bare (control): no residual or normalization
  2. Residual-only:  unconstrained residual connections at each interface
  3. Full:           gated residual + BatchNorm + higher LR for Pipeline B

All hybrid models use orthogonal initialization and soft orthogonality penalty
to isolate gradient-flow from symmetry-breaking.

Target: f(x,y) = ((x+y)^2+1)^3 + ((x-y)^2-1)^2

Pipeline A: square -> add_one -> cube   = ((proj_a)^2+1)^3
Pipeline B: square -> sub_one -> square = ((proj_b)^2-1)^2
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

PIPELINE_A_SPECS = [
    ("square",  "(* x x)"),
    ("add_one", "(+ x 1)"),
    ("cube",    "(let ((x2 (* x x))) (* x x2))"),
]

PIPELINE_B_SPECS = [
    ("square",  "(* x x)"),
    ("sub_one", "(- x 1)"),
    ("square2", "(* x x)"),
]


def target_fn(x, y):
    s_a = x + y
    s_b = x - y
    return (s_a**2 + 1) ** 3 + (s_b**2 - 1) ** 2


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
    print(f"  {label}: {' -> '.join(parts)} = {total_nodes} nodes, {total_edges} edges, {n_consts} consts")
    return subgraphs, total_nodes, total_edges, n_consts


def ortho_penalty(proj_a, proj_b):
    a = proj_a.weight.squeeze()
    b = proj_b.weight.squeeze()
    cos = torch.dot(a, b) / (a.norm() * b.norm() + 1e-8)
    return cos ** 2


class ResidualPipeline(nn.Module):
    def __init__(self, subgraphs):
        super().__init__()
        self.subgraphs = nn.ModuleList(subgraphs)
        n = len(subgraphs)
        self.residual_scale = nn.ParameterList([
            nn.Parameter(torch.tensor(0.1)) for _ in range(n)
        ])

    def forward(self, z):
        for i, sg in enumerate(self.subgraphs):
            z_new = sg.forward_batch({"x": z})
            z = z_new + self.residual_scale[i] * z
        return z


class GatedResidualBNPipeline(nn.Module):
    def __init__(self, subgraphs):
        super().__init__()
        self.subgraphs = nn.ModuleList(subgraphs)
        n = len(subgraphs)
        self.residual_gate = nn.ParameterList([
            nn.Parameter(torch.tensor(0.0)) for _ in range(n)
        ])
        self.norms = nn.ModuleList([
            nn.BatchNorm1d(1, momentum=0.01) for _ in range(n)
        ])

    def forward(self, z):
        for i, sg in enumerate(self.subgraphs):
            z_new = sg.forward_batch({"x": z})
            alpha = torch.sigmoid(self.residual_gate[i])
            z = z_new + alpha * z
            z = self.norms[i](z.unsqueeze(1)).squeeze(1)
        return z


class HybridModel(nn.Module):
    def __init__(self, pipeline_a, pipeline_b, is_bare=False):
        super().__init__()
        self.proj_a = nn.Linear(2, 1, bias=False)
        self.proj_b = nn.Linear(2, 1, bias=False)
        self.is_bare = is_bare
        if is_bare:
            self.pipeline_a = nn.ModuleList(pipeline_a)
            self.pipeline_b = nn.ModuleList(pipeline_b)
        else:
            self.pipeline_a = pipeline_a
            self.pipeline_b = pipeline_b
        self.output = nn.Linear(4, 1)

    def forward(self, x, y):
        inp = torch.stack([x, y], dim=1)
        z_a = self.proj_a(inp).squeeze(1)
        if self.is_bare:
            for sg in self.pipeline_a:
                z_a = sg.forward_batch({"x": z_a})
        else:
            z_a = self.pipeline_a(z_a)

        z_b = self.proj_b(inp).squeeze(1)
        if self.is_bare:
            for sg in self.pipeline_b:
                z_b = sg.forward_batch({"x": z_b})
        else:
            z_b = self.pipeline_b(z_b)

        features = torch.stack([z_a, z_b, x, y], dim=1)
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


def init_projections(model, a_init, b_init):
    with torch.no_grad():
        model.proj_a.weight.copy_(a_init)
        model.proj_b.weight.copy_(b_init)


def get_beta(model):
    return model.proj_b.weight.detach().squeeze().norm().item()


def get_dir_cos(model):
    w = model.proj_b.weight.detach().squeeze()
    true_dir = torch.tensor([1.0, -1.0]) / 2**0.5
    return abs(torch.dot(w / (w.norm() + 1e-8), true_dir)).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--train-range", type=float, default=1.0)
    parser.add_argument("--extrap-range", type=float, default=2.0)
    parser.add_argument("--ortho-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 70)
    print("EXPERIMENT 8: Residual Deep Composition — Ablation Study")
    print("=" * 70)
    print(f"\nTarget: f(x,y) = ((x+y)^2+1)^3 + ((x-y)^2-1)^2")
    print(f"Pipeline A: square -> add_one -> cube")
    print(f"Pipeline B: square -> sub_one -> square  (has gradient trap)")

    print("\nCompiling pipelines...")
    sgs_bare_a, na, ea, ca = compile_pipeline(PIPELINE_A_SPECS, "Pipeline A")
    sgs_bare_b, nb, eb, cb = compile_pipeline(PIPELINE_B_SPECS, "Pipeline B")
    sgs_res_a, _, _, _ = compile_pipeline(PIPELINE_A_SPECS, "Pipeline A (resid)")
    sgs_res_b, _, _, _ = compile_pipeline(PIPELINE_B_SPECS, "Pipeline B (resid)")
    sgs_full_a, _, _, _ = compile_pipeline(PIPELINE_A_SPECS, "Pipeline A (full)")
    sgs_full_b, _, _, _ = compile_pipeline(PIPELINE_B_SPECS, "Pipeline B (full)")

    print(f"\n  Frozen graph per model: {na + nb} nodes, {ea + eb} edges, {ca + cb} consts")

    # Build models
    bare = HybridModel(sgs_bare_a, sgs_bare_b, is_bare=True)
    resid = HybridModel(ResidualPipeline(sgs_res_a), ResidualPipeline(sgs_res_b))
    full = HybridModel(GatedResidualBNPipeline(sgs_full_a), GatedResidualBNPipeline(sgs_full_b))
    mlp = PureMLPModel(hidden=64, layers=3)

    # Same orthogonal init for all hybrid models
    proj_a_init = torch.tensor([[0.5, 0.5]])
    proj_b_init = torch.tensor([[0.5, -0.5]])
    for m in [bare, resid, full]:
        init_projections(m, proj_a_init.clone(), proj_b_init.clone())

    np_bare = sum(p.numel() for p in bare.parameters() if p.requires_grad)
    np_resid = sum(p.numel() for p in resid.parameters() if p.requires_grad)
    np_full = sum(p.numel() for p in full.parameters() if p.requires_grad)
    np_mlp = sum(p.numel() for p in mlp.parameters())

    print(f"\nTrainable parameters:")
    print(f"  Bare:          {np_bare}")
    print(f"  Residual-only: {np_resid}  (+{np_resid - np_bare} residual scales)")
    print(f"  Full:          {np_full}  (+gated residuals, +BN, +split LR)")
    print(f"  MLP:           {np_mlp}")
    print(f"\nInit: proj_a=[0.5, 0.5], proj_b=[0.5, -0.5]")

    # Optimizers — full model gets 10x LR for Pipeline B params
    opt_bare = torch.optim.Adam(bare.parameters(), lr=args.lr)
    opt_resid = torch.optim.Adam(resid.parameters(), lr=args.lr)

    full_pipe_b_params = list(full.pipeline_b.parameters()) + [full.proj_b.weight]
    full_other_params = [p for p in full.parameters() if not any(p is q for q in full_pipe_b_params)]
    opt_full = torch.optim.Adam([
        {"params": full_other_params, "lr": args.lr},
        {"params": full_pipe_b_params, "lr": args.lr * 10},
    ])

    opt_mlp = torch.optim.Adam(mlp.parameters(), lr=args.lr)

    R = args.train_range
    models = {"bare": bare, "resid": resid, "full": full, "mlp": mlp}
    optimizers = {"bare": opt_bare, "resid": opt_resid, "full": opt_full, "mlp": opt_mlp}
    histories = {k: [] for k in models}
    beta_histories = {k: [] for k in ["bare", "resid", "full"]}
    dir_histories = {k: [] for k in ["bare", "resid", "full"]}

    print(f"\nTraining ({args.epochs} epochs, batch={args.batch_size}, "
          f"lr={args.lr}, range [-{R}, {R}])...")
    print("-" * 70)

    for epoch in range(args.epochs):
        x = torch.empty(args.batch_size).uniform_(-R, R)
        y = torch.empty(args.batch_size).uniform_(-R, R)
        t = target_fn(x, y)

        for name, model in models.items():
            opt = optimizers[name]
            opt.zero_grad()
            pred = model(x, y)
            mse = nn.functional.mse_loss(pred, t)

            if name in ("bare", "resid", "full"):
                loss = mse + args.ortho_weight * ortho_penalty(model.proj_a, model.proj_b)
            else:
                loss = mse

            loss.backward()
            opt.step()
            histories[name].append(mse.item())

        with torch.no_grad():
            for name in ["bare", "resid", "full"]:
                beta_histories[name].append(get_beta(models[name]))
                dir_histories[name].append(get_dir_cos(models[name]))

        if epoch % 2000 == 0 or epoch == args.epochs - 1:
            betas = "  ".join(f"{k}={get_beta(models[k]):.3f}" for k in ["bare", "resid", "full"])
            mses = "  ".join(f"{k}={histories[k][-1]:.4f}" for k in models)
            print(f"  Epoch {epoch:5d}: {mses}")
            print(f"              beta: {betas}")

    # --- Results ---
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    for name in ["bare", "resid", "full"]:
        m = models[name]
        pa = m.proj_a.weight.detach().squeeze().tolist()
        pb = m.proj_b.weight.detach().squeeze().tolist()
        ow = m.output.weight.detach().squeeze().tolist()
        ob = m.output.bias.detach().item()
        beta = get_beta(m)
        dcos = get_dir_cos(m)

        print(f"\n  [{name.upper()}]")
        print(f"    proj_a: [{pa[0]:+.4f}, {pa[1]:+.4f}]  |w|={sum(x**2 for x in pa)**0.5:.4f}")
        print(f"    proj_b: [{pb[0]:+.4f}, {pb[1]:+.4f}]  |w|={beta:.4f}  cos(true)={dcos:.4f}")
        print(f"    output: [{ow[0]:+.4f}, {ow[1]:+.4f}, {ow[2]:+.4f}, {ow[3]:+.4f}]  bias={ob:+.4f}")

        if name == "resid" and hasattr(m.pipeline_b, "residual_scale"):
            rs = [p.item() for p in m.pipeline_b.residual_scale]
            print(f"    resid_b scales: [{', '.join(f'{v:+.4f}' for v in rs)}]")
        elif name == "full" and hasattr(m.pipeline_b, "residual_gate"):
            gates = [torch.sigmoid(p).item() for p in m.pipeline_b.residual_gate]
            print(f"    resid_b gates (sigmoid): [{', '.join(f'{v:.4f}' for v in gates)}]")

    # Test accuracy
    for name in models:
        models[name].eval()

    with torch.no_grad():
        x_t = torch.empty(10000).uniform_(-R, R)
        y_t = torch.empty(10000).uniform_(-R, R)
        t_t = target_fn(x_t, y_t)
        mse_in = {k: nn.functional.mse_loss(models[k](x_t, y_t), t_t).item() for k in models}

    E = args.extrap_range
    with torch.no_grad():
        x_e = torch.empty(10000).uniform_(R, E)
        y_e = torch.empty(10000).uniform_(R, E)
        t_e = target_fn(x_e, y_e)
        mse_ex = {k: nn.functional.mse_loss(models[k](x_e, y_e), t_e).item() for k in models}

    print(f"\n  In-distribution MSE [-{R}, {R}]:")
    for k in models:
        print(f"    {k:8s}: {mse_in[k]:.6f}")

    print(f"\n  Extrapolation MSE [{R}, {E}]:")
    for k in models:
        print(f"    {k:8s}: {mse_ex[k]:.6f}")

    # Gradient analysis
    print(f"\n  Gradient norms (at convergence):")
    for name in ["bare", "resid", "full"]:
        m = models[name]
        m.train()
        m.zero_grad()
        x_g = torch.empty(512).uniform_(-R, R)
        y_g = torch.empty(512).uniform_(-R, R)
        t_g = target_fn(x_g, y_g)
        nn.functional.mse_loss(m(x_g, y_g), t_g).backward()
        gpa = m.proj_a.weight.grad.norm().item()
        gpb = m.proj_b.weight.grad.norm().item()
        print(f"    {name:8s}: proj_a={gpa:.6f}  proj_b={gpb:.6f}  ratio_b/a={gpb/(gpa+1e-10):.3f}")

    # Success assessment
    print("\n" + "=" * 70)
    print("ASSESSMENT")
    print("=" * 70)

    b_bare = get_beta(bare)
    b_resid = get_beta(resid)
    b_full = get_beta(full)

    print(f"\n  proj_b magnitude (beta):")
    print(f"    Bare:     {b_bare:.4f}  {'TRAPPED' if b_bare < 0.5 else 'OK'}")
    print(f"    Residual: {b_resid:.4f}  {'TRAPPED' if b_resid < 0.5 else 'OK'}")
    print(f"    Full:     {b_full:.4f}  {'TRAPPED' if b_full < 0.5 else 'OK'}")

    print(f"\n  In-dist MSE improvement over bare:")
    if mse_in["bare"] > 0:
        print(f"    Residual: {mse_in['bare']/mse_in['resid']:.1f}x" if mse_in["resid"] > 0 else "    Residual: inf")
        print(f"    Full:     {mse_in['bare']/mse_in['full']:.1f}x" if mse_in["full"] > 0 else "    Full: inf")

    print(f"\n  Extrapolation MSE improvement over bare:")
    if mse_ex["bare"] > 0:
        print(f"    Residual: {mse_ex['bare']/mse_ex['resid']:.1f}x" if mse_ex["resid"] > 0 else "    Residual: inf")
        print(f"    Full:     {mse_ex['bare']/mse_ex['full']:.1f}x" if mse_ex["full"] > 0 else "    Full: inf")

    best_hybrid = min(["bare", "resid", "full"], key=lambda k: mse_in[k])
    if mse_in[best_hybrid] < 0.01:
        print(f"\n  >>> SUCCESS: {best_hybrid} converged (MSE={mse_in[best_hybrid]:.6f}) <<<")
    elif mse_in[best_hybrid] < mse_in["bare"] * 0.5:
        print(f"\n  >>> PARTIAL: {best_hybrid} improved {mse_in['bare']/mse_in[best_hybrid]:.1f}x "
              f"but MSE={mse_in[best_hybrid]:.4f} <<<")
    else:
        print(f"\n  >>> Sub_one attractor dominates all variants. <<<")
        print(f"      Best hybrid MSE: {mse_in[best_hybrid]:.4f} ({best_hybrid})")
        print(f"      This confirms the depth-optimization tradeoff finding from Exp 7.")

    # --- Plot ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "Experiment 8: Residual Connections at Frozen Subgraph Interfaces — Ablation",
        fontsize=13,
    )

    colors = {"bare": "#e74c3c", "resid": "#2ecc71", "full": "#9b59b6", "mlp": "#3498db"}
    labels = {"bare": "Bare", "resid": "Residual", "full": "Full (resid+BN+LR)", "mlp": "MLP"}

    # Training loss (smoothed)
    ax = axes[0, 0]
    win = 200
    for k in models:
        h = np.array(histories[k])
        smooth = np.convolve(h, np.ones(win) / win, mode="valid")
        np_k = {"bare": np_bare, "resid": np_resid, "full": np_full, "mlp": np_mlp}[k]
        ax.semilogy(smooth, alpha=0.8, label=f"{labels[k]} ({np_k}p)", color=colors[k])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (log)")
    ax.set_title("Training Loss (smoothed)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Beta trajectory
    ax = axes[0, 1]
    for k in ["bare", "resid", "full"]:
        ax.plot(beta_histories[k], alpha=0.8, label=labels[k], color=colors[k])
    ax.axhline(y=2**0.5, color="black", linestyle="--", alpha=0.4, label="|[1,-1]| canonical")
    ax.axhline(y=0.3, color="gray", linestyle=":", alpha=0.4, label="Exp 7 trap")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("||proj_b||")
    ax.set_title("Pipeline B Projection Magnitude")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Direction fidelity
    ax = axes[0, 2]
    for k in ["bare", "resid", "full"]:
        ax.plot(dir_histories[k], alpha=0.8, label=labels[k], color=colors[k])
    ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("cos(proj_b, [1,-1])")
    ax.set_title("Pipeline B Direction Fidelity")
    ax.legend(fontsize=7)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # Projection weights
    ax = axes[1, 0]
    bar_labels = ["pA[0]", "pA[1]", "pB[0]", "pB[1]"]
    true_v = [1, 1, 1, -1]
    x_pos = np.arange(len(bar_labels))
    width = 0.18
    for i, k in enumerate(["bare", "resid", "full"]):
        m = models[k]
        pa_v = m.proj_a.weight.detach().squeeze().tolist()
        pb_v = m.proj_b.weight.detach().squeeze().tolist()
        vals = [pa_v[0], pa_v[1], pb_v[0], pb_v[1]]
        ax.bar(x_pos + (i - 1) * width, vals, width, label=labels[k],
               color=colors[k], edgecolor="black", linewidth=0.5)
    ax.bar(x_pos + 2 * width, true_v, width, label="Canonical",
           color="#95a5a6", alpha=0.5, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bar_labels)
    ax.set_ylabel("Weight")
    ax.set_title("Projection Weights")
    ax.legend(fontsize=7)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    # MSE comparison bar chart
    ax = axes[1, 1]
    bar_models = list(models.keys())
    x_b = np.arange(len(bar_models))
    in_vals = [mse_in[k] for k in bar_models]
    ex_vals = [mse_ex[k] for k in bar_models]
    width_b = 0.35
    ax.bar(x_b - width_b / 2, in_vals, width_b, label="In-distribution",
           color="#3498db", edgecolor="black", linewidth=0.5)
    ax.bar(x_b + width_b / 2, ex_vals, width_b, label="Extrapolation",
           color="#e67e22", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x_b)
    ax.set_xticklabels([labels[k] for k in bar_models], fontsize=8)
    ax.set_ylabel("MSE (log)")
    ax.set_yscale("log")
    ax.set_title("Test MSE Comparison")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # Function comparison
    ax = axes[1, 2]
    xp = torch.linspace(-E, E, 500)
    yf = torch.full_like(xp, 0.5)
    for k in models:
        models[k].eval()
    with torch.no_grad():
        tp = target_fn(xp, yf)
        preds = {k: models[k](xp, yf) for k in models}
    ax.plot(xp, tp, "k-", linewidth=2, label="True f(x, 0.5)")
    for k in ["resid", "full", "bare"]:
        ax.plot(xp, preds[k], "--" if k != "bare" else ":", linewidth=1.5,
                label=labels[k], color=colors[k], alpha=0.8)
    ax.plot(xp, preds["mlp"], ":", linewidth=1, label="MLP", color=colors["mlp"], alpha=0.5)
    ax.axvspan(-R, R, alpha=0.1, color="blue", label="Train range")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x, 0.5)")
    ax.set_title("Function Fit: f(x, 0.5)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "hybrid_residual_composition.png"
    )
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to {fig_path}")


if __name__ == "__main__":
    main()

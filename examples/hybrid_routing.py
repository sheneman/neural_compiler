############################################################
#
# Neural Compiler: Compiling scheme into composable and
#                  differentiable neural network representations
#
# hybrid_routing.py: Trainable router learning to select among frozen compiled subgraphs
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Hybrid routing: trainable router + frozen compiled modules.

Demonstrates a trainable neural network that learns to route inputs to
deterministically compiled Scheme programs embedded as frozen compiled modules.

The target is a piecewise function:
    f(x) = x² + 1       for x < -1
    f(x) = 2x - 3       for -1 ≤ x ≤ 2
    f(x) = -x² + 3x     for x > 2

Three Scheme programs are compiled to frozen compiled modules. A small trainable
MLP learns the routing weights (which subgraph to use for each input).
"""

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_compiler.compiler import compile_scheme
from neural_compiler.evaluator.direct_module import DirectModule

PROGRAMS = [
    ("(+ (* x x) 1)", "x² + 1"),
    ("(- (* 2 x) 3)", "2x - 3"),
    ("(+ (- 0 (* x x)) (* 3 x))", "-x² + 3x"),
]


def target_function(x: torch.Tensor) -> torch.Tensor:
    y = torch.zeros_like(x)
    m1 = x < -1
    m2 = (x >= -1) & (x <= 2)
    m3 = x > 2
    y[m1] = x[m1] ** 2 + 1
    y[m2] = 2 * x[m2] - 3
    y[m3] = -x[m3] ** 2 + 3 * x[m3]
    return y


def compile_subgraphs() -> list[DirectModule]:
    models = []
    for source, label in PROGRAMS:
        graph = compile_scheme(source, inputs={"x": None})
        model = DirectModule(graph)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        models.append(model)
        print(f"  Compiled '{label}': {len(graph.nodes)} nodes, depth={graph.depth()}")
    return models


class HybridPiecewiseModel(nn.Module):
    def __init__(self, subgraphs: list[DirectModule]):
        super().__init__()
        self.subgraphs = nn.ModuleList(subgraphs)
        self.temperature = 1.0
        self.router = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, len(subgraphs)),
        )

    def forward(self, x: torch.Tensor):
        outputs = torch.stack(
            [sg.forward_batch({"x": x}) for sg in self.subgraphs], dim=1
        )
        logits = self.router(x.unsqueeze(1))
        weights = F.softmax(logits / self.temperature, dim=1)
        result = (weights * outputs).sum(dim=1)
        return result, weights


class PureMLPBaseline(nn.Module):
    def __init__(self, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x.unsqueeze(1)).squeeze(1)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_frozen_structure(subgraphs: list[DirectModule]) -> dict:
    total_nodes = 0
    total_edges = 0
    total_consts = 0
    for sg in subgraphs:
        g = sg.graph
        total_nodes += len(g.nodes)
        total_consts += sum(1 for n in g.nodes.values() if n.op_type == "const")
        total_edges += sum(
            s.edge_index.shape[1] for s in sg._data_template.edge_stores
            if hasattr(s, "edge_index")
        )
        for lb in sg._loop_bodies.values():
            total_nodes += len(lb.body_graph.nodes)
            total_consts += sum(1 for n in lb.body_graph.nodes.values() if n.op_type == "const")
            total_edges += sum(
                s.edge_index.shape[1] for s in lb.data_template.edge_stores
                if hasattr(s, "edge_index")
            )
    return {"nodes": total_nodes, "edges": total_edges, "consts": total_consts}


def train(model, is_hybrid, epochs, batch_size, lr):
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    losses = []
    for epoch in range(epochs):
        if is_hybrid:
            t = epoch / max(epochs - 1, 1)
            model.temperature = 1.0 - 0.8 * t  # anneal 1.0 → 0.2

        x = torch.FloatTensor(batch_size).uniform_(-5, 5)
        y = target_function(x)

        if is_hybrid:
            y_pred, _ = model(x)
        else:
            y_pred = model(x)

        loss = F.mse_loss(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if epoch % 500 == 0 or epoch == epochs - 1:
            extra = f"  temp={model.temperature:.2f}" if is_hybrid else ""
            print(f"  Epoch {epoch:>5d}: loss = {loss.item():.6f}{extra}")
    return losses


def visualize(hybrid, mlp, h_losses, m_losses, save_path):
    x_eval = torch.linspace(-7, 7, 1000)
    y_true = target_function(x_eval)

    with torch.no_grad():
        sg_out = [sg.forward_batch({"x": x_eval}).numpy() for sg in hybrid.subgraphs]
        y_hyb, weights = hybrid(x_eval)
        y_hyb = y_hyb.numpy()
        weights = weights.numpy()
        y_mlp = mlp(x_eval).numpy()

    x_np = x_eval.numpy()
    y_np = y_true.numpy()

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), height_ratios=[3, 2, 2])

    # --- Panel 1: function fits ---
    ax = axes[0]
    colors = ["#2196F3", "#FF9800", "#4CAF50"]
    for i, (_, label) in enumerate(PROGRAMS):
        ax.plot(x_np, sg_out[i], "--", color=colors[i], alpha=0.5, label=f"Subgraph: {label}")
    ax.plot(x_np, y_np, "k-", linewidth=2.5, label="True f(x)")
    ax.plot(x_np, y_hyb, "r-", linewidth=2, label="Hybrid model")
    ax.plot(x_np, y_mlp, "b-", linewidth=2, alpha=0.7, label="Pure MLP")
    ax.axvline(-1, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(2, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlim(-7, 7)
    ax.set_ylim(-15, 30)
    ax.set_ylabel("f(x)")
    ax.set_title("Neural Compiler: Hybrid Routing with Compiled Subgraphs")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: routing weights ---
    ax = axes[1]
    for i, (_, label) in enumerate(PROGRAMS):
        ax.plot(x_np, weights[:, i], color=colors[i], linewidth=2, label=f"w({label})")
    ax.axvline(-1, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(2, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlim(-7, 7)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Routing weight")
    ax.set_title("Learned Routing Weights")
    ax.legend(loc="center right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel 3: loss curves ---
    ax = axes[2]
    window = 50
    h_smooth = np.convolve(h_losses, np.ones(window) / window, mode="valid")
    m_smooth = np.convolve(m_losses, np.ones(window) / window, mode="valid")
    ax.semilogy(h_smooth, "r-", label="Hybrid model")
    ax.semilogy(m_smooth, "b-", alpha=0.7, label="Pure MLP")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (log)")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Hybrid routing example")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save-fig", default="examples/hybrid_routing.png")
    args = parser.parse_args()

    print("Compiling Scheme programs to frozen compiled modules...")
    subgraphs = compile_subgraphs()

    hybrid = HybridPiecewiseModel(subgraphs)
    mlp = PureMLPBaseline(hidden=16)
    frozen = count_frozen_structure(subgraphs)
    print(f"\nHybrid model trainable params: {count_params(hybrid)}")
    print(f"  Frozen subgraph structure: {frozen['nodes']} nodes, {frozen['edges']} edges, {frozen['consts']} const floats")
    print(f"Pure MLP trainable params:     {count_params(mlp)}")

    print(f"\nTraining hybrid model ({args.epochs} epochs)...")
    h_losses = train(hybrid, is_hybrid=True, epochs=args.epochs,
                     batch_size=args.batch_size, lr=args.lr)

    print(f"\nTraining pure MLP ({args.epochs} epochs)...")
    m_losses = train(mlp, is_hybrid=False, epochs=args.epochs,
                     batch_size=args.batch_size, lr=args.lr)

    x_test = torch.linspace(-5, 5, 10000)
    y_test = target_function(x_test)
    with torch.no_grad():
        y_hyb, _ = hybrid(x_test)
        y_mlp = mlp(x_test)
    h_mse = F.mse_loss(y_hyb, y_test).item()
    m_mse = F.mse_loss(y_mlp, y_test).item()
    print(f"\nTest MSE (x in [-5, 5]):")
    print(f"  Hybrid: {h_mse:.6f}")
    print(f"  MLP:    {m_mse:.6f}")
    print(f"  Ratio:  MLP is {m_mse / max(h_mse, 1e-12):.1f}x worse")

    x_extrap = torch.linspace(-10, 10, 10000)
    y_extrap = target_function(x_extrap)
    with torch.no_grad():
        y_hyb_e, _ = hybrid(x_extrap)
        y_mlp_e = mlp(x_extrap)
    h_mse_e = F.mse_loss(y_hyb_e, y_extrap).item()
    m_mse_e = F.mse_loss(y_mlp_e, y_extrap).item()
    print(f"\nExtrapolation MSE (x in [-10, 10]):")
    print(f"  Hybrid: {h_mse_e:.6f}")
    print(f"  MLP:    {m_mse_e:.6f}")
    print(f"  Ratio:  MLP is {m_mse_e / max(h_mse_e, 1e-12):.1f}x worse")

    visualize(hybrid, mlp, h_losses, m_losses, args.save_fig)


if __name__ == "__main__":
    main()

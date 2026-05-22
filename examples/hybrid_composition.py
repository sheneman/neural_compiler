############################################################
#
# Neural Compiler: Compiling scheme into composable and
#                  differentiable neural network representations
#
# hybrid_composition.py: Two-stage composition with compiled subgraph outputs feeding learned weights
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Hybrid architecture with multi-stage composition of compiled subgraphs.

Demonstrates that compiled modules can be COMPOSED — outputs from
one set of subgraphs feed as inputs to another set — with gradient flow
through two frozen subgraphs in series.

Target function (hidden):
    f(x, y) = w1 * cubic(sum_sq(x, y)) + w2 * quadratic(diff_sq(x, y)) + bias

where:
    sum_sq(x, y) = x² + y²      (Stage 1)
    diff_sq(x, y) = x² - y²     (Stage 1)
    cubic(z) = z³                (Stage 2)
    quadratic(z) = z² - 1        (Stage 2)

True weights: w1=1.0, w2=-1.0, bias=0.5.

The model must discover:
  1. Which stage-1 output feeds each stage-2 subgraph (wiring)
  2. The output combination weights

Gradient flows through TWO frozen subgraphs in series:
  loss → output_weights → stage-2 frozen module → wiring projections → stage-1 outputs
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_compiler.compiler import compile_scheme
from neural_compiler.evaluator.direct_module import DirectModule

SUM_SQ_SRC = "(+ (* x x) (* y y))"
DIFF_SQ_SRC = "(- (* x x) (* y y))"
CUBIC_SRC = "(let ((x2 (* x x))) (* x x2))"
QUADRATIC_SRC = "(- (* x x) 1)"

TRUE_WEIGHTS = {"w_cubic": 1.0, "w_quad": -1.0, "bias": 0.5}


def target_function(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    s1 = x ** 2 + y ** 2
    s2 = x ** 2 - y ** 2
    o1 = s1 ** 3
    o2 = s2 ** 2 - 1
    return 1.0 * o1 + (-1.0) * o2 + 0.5


def compile_subgraphs():
    graph_ss = compile_scheme(SUM_SQ_SRC, inputs={"x": None, "y": None})
    model_ss = DirectModule(graph_ss)
    for p in model_ss.parameters():
        p.requires_grad = False
    model_ss.eval()
    print(f"  Compiled sum_sq(x,y): {len(graph_ss.nodes)} nodes")

    graph_ds = compile_scheme(DIFF_SQ_SRC, inputs={"x": None, "y": None})
    model_ds = DirectModule(graph_ds)
    for p in model_ds.parameters():
        p.requires_grad = False
    model_ds.eval()
    print(f"  Compiled diff_sq(x,y): {len(graph_ds.nodes)} nodes")

    graph_cub = compile_scheme(CUBIC_SRC, inputs={"x": None})
    model_cub = DirectModule(graph_cub)
    for p in model_cub.parameters():
        p.requires_grad = False
    model_cub.eval()
    print(f"  Compiled cubic(z): {len(graph_cub.nodes)} nodes")

    graph_quad = compile_scheme(QUADRATIC_SRC, inputs={"x": None})
    model_quad = DirectModule(graph_quad)
    for p in model_quad.parameters():
        p.requires_grad = False
    model_quad.eval()
    print(f"  Compiled quadratic(z): {len(graph_quad.nodes)} nodes")

    return model_ss, model_ds, model_cub, model_quad


class HybridCompositionModel(nn.Module):
    def __init__(self, sg_sum_sq, sg_diff_sq, sg_cubic, sg_quadratic):
        super().__init__()
        self.sg_sum_sq = sg_sum_sq
        self.sg_diff_sq = sg_diff_sq
        self.sg_cubic = sg_cubic
        self.sg_quadratic = sg_quadratic

        # Wiring: each stage-2 subgraph selects from stage-1 outputs via learned projection
        self.proj_cubic = nn.Linear(2, 1, bias=False)
        self.proj_quad = nn.Linear(2, 1, bias=False)

        # Output combination
        self.output_weights = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        # Stage 1: compute both subgraphs on raw inputs (no grad needed —
        # no trainable params before stage 1)
        with torch.no_grad():
            s1 = self.sg_sum_sq.forward_batch({"x": x, "y": y})
            s2 = self.sg_diff_sq.forward_batch({"x": x, "y": y})

        # Learned wiring: project stage-1 outputs to stage-2 inputs
        stage1_out = torch.stack([s1, s2], dim=1)
        z_cubic = self.proj_cubic(stage1_out).squeeze(1)
        z_quad = self.proj_quad(stage1_out).squeeze(1)

        # Stage 2: feed through frozen subgraphs (grad MUST flow for wiring to learn)
        o_cubic = self.sg_cubic.forward_batch({"x": z_cubic})
        o_quad = self.sg_quadratic.forward_batch({"x": z_quad})

        # Output combination
        features = torch.stack([o_cubic, o_quad], dim=1)
        return self.output_weights(features).squeeze(1)


class PureMLPBaseline(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        xy = torch.stack([x, y], dim=1)
        return self.net(xy).squeeze(1)


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
        if hasattr(sg, "_loop_bodies"):
            for lb in sg._loop_bodies.values():
                total_nodes += len(lb.body_graph.nodes)
                total_consts += sum(
                    1 for n in lb.body_graph.nodes.values() if n.op_type == "const"
                )
                total_edges += sum(
                    s.edge_index.shape[1]
                    for s in lb.data_template.edge_stores
                    if hasattr(s, "edge_index")
                )
    return {"nodes": total_nodes, "edges": total_edges, "consts": total_consts}


def generate_batch(batch_size, xy_range=1.0):
    x = torch.FloatTensor(batch_size).uniform_(-xy_range, xy_range)
    y = torch.FloatTensor(batch_size).uniform_(-xy_range, xy_range)
    z = target_function(x, y)
    return x, y, z


def train(model, epochs, batch_size, lr):
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    losses = []
    for epoch in range(epochs):
        x, y, z = generate_batch(batch_size)
        z_pred = model(x, y)
        loss = F.mse_loss(z_pred, z)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch % 500 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch:>5d}: loss = {loss.item():.6f}")
    return losses


def visualize(hybrid, mlp, h_losses, m_losses, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Loss curves
    ax = axes[0, 0]
    window = 30
    h_smooth = np.convolve(h_losses, np.ones(window) / window, mode="valid")
    m_smooth = np.convolve(m_losses, np.ones(window) / window, mode="valid")
    ax.semilogy(h_smooth, "r-", linewidth=2, label="Hybrid (composition)")
    ax.semilogy(m_smooth, "b-", linewidth=2, alpha=0.7, label="Pure MLP")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (log)")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Learned wiring + output weights
    ax = axes[0, 1]
    pc = hybrid.proj_cubic.weight.data[0].tolist()
    pq = hybrid.proj_quad.weight.data[0].tolist()
    ow = hybrid.output_weights.weight.data[0].tolist()
    ob = hybrid.output_weights.bias.data[0].item()

    labels = ["p_cub[s1]", "p_cub[s2]", "p_qd[s1]", "p_qd[s2]",
              "w_cub", "w_qd", "bias"]
    learned = [pc[0], pc[1], pq[0], pq[1], ow[0], ow[1], ob]
    true = [1.0, 0.0, 0.0, 1.0, 1.0, -1.0, 0.5]
    x_pos = np.arange(len(labels))
    ax.bar(x_pos - 0.15, learned, 0.3, label="Learned", color="#2196F3", alpha=0.8)
    ax.bar(x_pos + 0.15, true, 0.3, label="True (canonical)", color="#FF9800", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=8, rotation=30)
    ax.set_ylabel("Weight value")
    ax.set_title("Learned Wiring & Combination Weights")
    ax.legend()
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 3: 1D slice along x (y=0.5 fixed)
    ax = axes[1, 0]
    x_range = torch.linspace(-2, 2, 300)
    y_fixed = torch.full_like(x_range, 0.5)
    z_true = target_function(x_range, y_fixed).numpy()
    with torch.no_grad():
        z_hyb = hybrid(x_range, y_fixed).numpy()
        z_mlp = mlp(x_range, y_fixed).numpy()
    ax.plot(x_range.numpy(), z_true, "k-", linewidth=2.5, label="True f(x, 0.5)")
    ax.plot(x_range.numpy(), z_hyb, "r--", linewidth=2, label="Hybrid")
    ax.plot(x_range.numpy(), z_mlp, "b--", linewidth=2, alpha=0.7, label="MLP")
    ax.axvline(-1, color="gray", linestyle=":", alpha=0.3)
    ax.axvline(1, color="gray", linestyle=":", alpha=0.3)
    ax.set_xlabel("x (y=0.5)")
    ax.set_ylabel("f(x, 0.5)")
    ax.set_title("Slice: f(x, y=0.5) with extrapolation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 4: 1D slice along y (x=0.5 fixed)
    ax = axes[1, 1]
    y_range = torch.linspace(-2, 2, 300)
    x_fixed = torch.full_like(y_range, 0.5)
    z_true = target_function(x_fixed, y_range).numpy()
    with torch.no_grad():
        z_hyb = hybrid(x_fixed, y_range).numpy()
        z_mlp = mlp(x_fixed, y_range).numpy()
    ax.plot(y_range.numpy(), z_true, "k-", linewidth=2.5, label="True f(0.5, y)")
    ax.plot(y_range.numpy(), z_hyb, "r--", linewidth=2, label="Hybrid")
    ax.plot(y_range.numpy(), z_mlp, "b--", linewidth=2, alpha=0.7, label="MLP")
    ax.axvline(-1, color="gray", linestyle=":", alpha=0.3)
    ax.axvline(1, color="gray", linestyle=":", alpha=0.3)
    ax.set_xlabel("y (x=0.5)")
    ax.set_ylabel("f(0.5, y)")
    ax.set_title("Slice: f(0.5, y) with extrapolation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Neural Compiler: Multi-Stage Composition of Compiled Subgraphs",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--save-fig", default="examples/hybrid_composition.png")
    args = parser.parse_args()

    print("Compiling Scheme programs to frozen compiled modules...")
    sg_ss, sg_ds, sg_cub, sg_quad = compile_subgraphs()

    hybrid = HybridCompositionModel(sg_ss, sg_ds, sg_cub, sg_quad)
    mlp = PureMLPBaseline(hidden=64)
    frozen = count_frozen_structure([sg_ss, sg_ds, sg_cub, sg_quad])
    print(f"\nHybrid trainable params: {count_params(hybrid)}")
    print(
        f"  Frozen subgraph structure: {frozen['nodes']} nodes, "
        f"{frozen['edges']} edges, {frozen['consts']} const floats"
    )
    print(f"Pure MLP trainable params: {count_params(mlp)}")

    print(f"\nTraining hybrid model ({args.epochs} epochs)...")
    h_losses = train(hybrid, args.epochs, args.batch_size, args.lr)

    print(f"\nTraining pure MLP ({args.epochs} epochs)...")
    m_losses = train(mlp, args.epochs, args.batch_size, args.lr)

    # Report learned parameters
    print("\nLearned parameters:")
    pc = hybrid.proj_cubic.weight.data[0].tolist()
    pq = hybrid.proj_quad.weight.data[0].tolist()
    ow = hybrid.output_weights.weight.data[0].tolist()
    ob = hybrid.output_weights.bias.data[0].item()
    print(f"  proj_cubic:  [{pc[0]:.4f}, {pc[1]:.4f}]  (true: [1, 0])")
    print(f"  proj_quad:   [{pq[0]:.4f}, {pq[1]:.4f}]  (true: [0, 1])")
    print(f"  w_cubic={ow[0]:.4f}  w_quad={ow[1]:.4f}  bias={ob:.4f}")
    print(f"  True: w_cubic=1.0  w_quad=-1.0  bias=0.5")

    # In-distribution test
    h_mse = 0.0
    m_mse = 0.0
    for _ in range(10):
        x, y, z = generate_batch(5000)
        with torch.no_grad():
            h_mse += F.mse_loss(hybrid(x, y), z).item()
            m_mse += F.mse_loss(mlp(x, y), z).item()
    h_mse /= 10
    m_mse /= 10
    print(f"\nTest MSE (in-distribution, averaged over 10 runs):")
    print(f"  Hybrid: {h_mse:.6f}")
    print(f"  MLP:    {m_mse:.6f}")
    print(f"  Ratio:  MLP is {m_mse / max(h_mse, 1e-12):.1f}x worse")

    # Extrapolation test (2x range)
    h_ext = 0.0
    m_ext = 0.0
    for _ in range(10):
        x, y, z = generate_batch(5000, xy_range=2.0)
        with torch.no_grad():
            h_ext += F.mse_loss(hybrid(x, y), z).item()
            m_ext += F.mse_loss(mlp(x, y), z).item()
    h_ext /= 10
    m_ext /= 10
    print(f"\nTest MSE (extrapolation 2x range, averaged over 10 runs):")
    print(f"  Hybrid: {h_ext:.6f}")
    print(f"  MLP:    {m_ext:.6f}")
    print(f"  Ratio:  MLP is {m_ext / max(h_ext, 1e-12):.1f}x worse")

    visualize(hybrid, mlp, h_losses, m_losses, args.save_fig)


if __name__ == "__main__":
    main()

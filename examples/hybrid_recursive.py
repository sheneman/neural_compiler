############################################################
#
# Neural Compiler: Compiling scheme into composable and
#                  differentiable neural network representations
#
# hybrid_recursive.py: Batched execution of recursive compiled subgraph in hybrid architecture
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Hybrid architecture with a recursive compiled subgraph (factorial).

Demonstrates that compiled modules can include recursive programs
(via TCO -> loop/recur) with batched execution in a hybrid trainable
architecture.

Target function (hidden):
    f(n, x) = w1 * factorial(n) + w2 * x^2 + w3 * x + w4

where n in {0, 1, ..., 7} and x in [-2, 2]. True weights: w1=0.01, w2=1, w3=2, w4=-1.

The scaling factor on factorial (0.01) tames the dynamic range while still
making the factorial term significant: 0.01 * 7! = 50.4, comparable to
the polynomial terms.

Compiled subgraphs:
    1. factorial(n) — recursive, compiled via TCO to loop/recur
    2. x^2 — simple DAG

The trainable model receives the subgraph outputs plus x and learns
the linear combination weights.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_compiler.compiler import compile_scheme
from neural_compiler.evaluator.direct_module import DirectModule

FACTORIAL_SRC = """
(letrec ((fact (lambda (n acc)
    (if (= n 0) acc (fact (- n 1) (* acc n))))))
  (fact n 1))
"""
SQUARE_SRC = "(* x x)"

TRUE_WEIGHTS = {"w_fact": 0.01, "w_sq": 1.0, "w_x": 2.0, "bias": -1.0}


def target_function(n: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    fact_vals = torch.ones_like(n)
    for i in range(len(n)):
        v = 1.0
        for j in range(1, int(n[i].item()) + 1):
            v *= j
        fact_vals[i] = v
    return 0.01 * fact_vals + x ** 2 + 2 * x - 1


def compile_subgraphs():
    graph_fact = compile_scheme(FACTORIAL_SRC, inputs={"n": None})
    model_fact = DirectModule(graph_fact)
    for p in model_fact.parameters():
        p.requires_grad = False
    model_fact.eval()
    print(f"  Compiled factorial: {len(graph_fact.nodes)} nodes, has_loops={graph_fact.has_loops}")

    graph_sq = compile_scheme(SQUARE_SRC, inputs={"x": None})
    model_sq = DirectModule(graph_sq)
    for p in model_sq.parameters():
        p.requires_grad = False
    model_sq.eval()
    print(f"  Compiled x^2: {len(graph_sq.nodes)} nodes")

    return model_fact, model_sq


class HybridRecursiveModel(nn.Module):
    def __init__(self, sg_factorial: DirectModule, sg_square: DirectModule):
        super().__init__()
        self.sg_factorial = sg_factorial
        self.sg_square = sg_square
        self.weights = nn.Linear(3, 1)

    def forward(self, n: torch.Tensor, x: torch.Tensor):
        n_int = torch.round(torch.clamp(n, 0, 7))
        o_fact = self.sg_factorial.forward_batch({"n": n_int})
        o_sq = self.sg_square.forward_batch({"x": x})
        features = torch.stack([o_fact, o_sq, x], dim=1)
        return self.weights(features).squeeze(1)


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

    def forward(self, n: torch.Tensor, x: torch.Tensor):
        nx = torch.stack([n, x], dim=1)
        return self.net(nx).squeeze(1)


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


def generate_batch(batch_size, x_range=2.0, n_max=7):
    n = torch.randint(0, n_max + 1, (batch_size,)).float()
    x = torch.FloatTensor(batch_size).uniform_(-x_range, x_range)
    y = target_function(n, x)
    return n, x, y


def train(model, epochs, batch_size, lr):
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    losses = []
    for epoch in range(epochs):
        n, x, y = generate_batch(batch_size)
        y_pred = model(n, x)
        loss = F.mse_loss(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch % 500 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch:>5d}: loss = {loss.item():.6f}")
    return losses


def visualize(hybrid, mlp, h_losses, m_losses, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Panel 1: Loss curves ---
    ax = axes[0, 0]
    window = 30
    h_smooth = np.convolve(h_losses, np.ones(window) / window, mode="valid")
    m_smooth = np.convolve(m_losses, np.ones(window) / window, mode="valid")
    ax.semilogy(h_smooth, "r-", linewidth=2, label="Hybrid (factorial + x²)")
    ax.semilogy(m_smooth, "b-", linewidth=2, alpha=0.7, label="Pure MLP")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (log)")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Learned weights ---
    ax = axes[0, 1]
    w = hybrid.weights.weight.data[0].tolist()
    b = hybrid.weights.bias.data[0].item()
    labels = ["w(fact)", "w(x²)", "w(x)", "bias"]
    learned = [w[0], w[1], w[2], b]
    true = [0.01, 1.0, 2.0, -1.0]
    x_pos = np.arange(4)
    ax.bar(x_pos - 0.15, learned, 0.3, label="Learned", color="#2196F3", alpha=0.8)
    ax.bar(x_pos + 0.15, true, 0.3, label="True", color="#FF9800", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Weight value")
    ax.set_title("Learned Combination Weights")
    ax.legend()
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    # --- Panel 3: f(n, x=1) for each n ---
    ax = axes[1, 0]
    n_vals = torch.arange(0, 8).float()
    x_ones = torch.ones(8)
    y_true = target_function(n_vals, x_ones)
    with torch.no_grad():
        y_hyb = hybrid(n_vals, x_ones)
        y_mlp = mlp(n_vals, x_ones)
    w = 0.25
    ax.bar(n_vals.numpy() - w, y_true.numpy(), w, label="True", color="black", alpha=0.7)
    ax.bar(n_vals.numpy(), y_hyb.numpy(), w, label="Hybrid", color="red", alpha=0.7)
    ax.bar(n_vals.numpy() + w, y_mlp.numpy(), w, label="MLP", color="blue", alpha=0.7)
    ax.set_xlabel("n")
    ax.set_ylabel("f(n, x=1)")
    ax.set_title("f(n, x=1) — factorial term grows with n")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Panel 4: f(n=5, x) slice ---
    ax = axes[1, 1]
    x_range = torch.linspace(-3, 3, 300)
    n_fixed = torch.full_like(x_range, 5.0)
    y_true = target_function(n_fixed, x_range).numpy()
    with torch.no_grad():
        y_hyb = hybrid(n_fixed, x_range).numpy()
        y_mlp = mlp(n_fixed, x_range).numpy()
    ax.plot(x_range.numpy(), y_true, "k-", linewidth=2.5, label="True f(5, x)")
    ax.plot(x_range.numpy(), y_hyb, "r--", linewidth=2, label="Hybrid")
    ax.plot(x_range.numpy(), y_mlp, "b--", linewidth=2, alpha=0.7, label="MLP")
    ax.axvline(-2, color="gray", linestyle=":", alpha=0.3)
    ax.axvline(2, color="gray", linestyle=":", alpha=0.3)
    ax.set_xlabel("x (n=5)")
    ax.set_ylabel("f(5, x) = 1.2 + x² + 2x - 1")
    ax.set_title("Slice: f(5, x) with extrapolation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Neural Compiler: Hybrid Architecture with Recursive Compiled Subgraph (Factorial)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--save-fig", default="examples/hybrid_recursive.png")
    args = parser.parse_args()

    print("Compiling Scheme programs to frozen compiled modules...")
    sg_fact, sg_sq = compile_subgraphs()

    hybrid = HybridRecursiveModel(sg_fact, sg_sq)
    mlp = PureMLPBaseline(hidden=64)
    frozen = count_frozen_structure([sg_fact, sg_sq])
    print(f"\nHybrid trainable params: {count_params(hybrid)}")
    print(f"  Frozen subgraph structure: {frozen['nodes']} nodes, {frozen['edges']} edges, {frozen['consts']} const floats")
    print(f"Pure MLP trainable params: {count_params(mlp)}")

    print(f"\nTraining hybrid model ({args.epochs} epochs)...")
    h_losses = train(hybrid, args.epochs, args.batch_size, args.lr)

    print(f"\nTraining pure MLP ({args.epochs} epochs)...")
    m_losses = train(mlp, args.epochs, args.batch_size, args.lr)

    print("\nLearned weights:")
    w = hybrid.weights.weight.data[0].tolist()
    b = hybrid.weights.bias.data[0].item()
    print(f"  w(fact)={w[0]:.4f}  w(x²)={w[1]:.4f}  w(x)={w[2]:.4f}  bias={b:.4f}")
    print(f"  True: w(fact)=0.01  w(x²)=1.0  w(x)=2.0  bias=-1.0")

    h_mse = 0.0
    m_mse = 0.0
    for _ in range(10):
        n, x, y = generate_batch(5000)
        with torch.no_grad():
            h_mse += F.mse_loss(hybrid(n, x), y).item()
            m_mse += F.mse_loss(mlp(n, x), y).item()
    h_mse /= 10
    m_mse /= 10
    print(f"\nTest MSE (averaged over 10 runs):")
    print(f"  Hybrid: {h_mse:.6f}")
    print(f"  MLP:    {m_mse:.6f}")
    print(f"  Ratio:  MLP is {m_mse / max(h_mse, 1e-12):.1f}x worse")

    visualize(hybrid, mlp, h_losses, m_losses, args.save_fig)


if __name__ == "__main__":
    main()

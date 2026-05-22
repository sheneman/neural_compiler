############################################################
#
# Neural Compiler: Compiling scheme into composable and
#                  differentiable neural network representations
#
# hybrid_interfacing.py: Trainable neural network learning input projections for frozen compiled modules
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Hybrid interfacing: learned input projections + frozen compiled subgraphs.

Demonstrates a trainable neural network that learns to TRANSFORM inputs before
passing them to deterministically compiled modules. The model learns both
the correct input projections and output combination weights.

Target function (hidden):
    f(a,b,c,d) = p1(a+b) + 2*p2(a-c, b+d) - p3(c-d)

where:
    p1(x)   = x^2 + 1
    p2(x,y) = x*y - x + y
    p3(x)   = x^3

The trainable model must discover:
    - Linear1 should project (a,b,c,d) -> a+b
    - Linear2 should project (a,b,c,d) -> (a-c, b+d)
    - Linear3 should project (a,b,c,d) -> c-d
    - Output weights should be [1, 2, -1]
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
    ("(+ (* x x) 1)", {"x": None}, "x² + 1"),
    ("(+ (- (* x y) x) y)", {"x": None, "y": None}, "xy - x + y"),
    ("(let ((x2 (* x x))) (* x x2))", {"x": None}, "x³"),
]

TRUE_PROJECTIONS = {
    "p1": "a + b",
    "p2_x": "a - c",
    "p2_y": "b + d",
    "p3": "c - d",
}
TRUE_OUTPUT_WEIGHTS = [1.0, 2.0, -1.0]


def target_function(abcd: torch.Tensor) -> torch.Tensor:
    a, b, c, d = abcd[:, 0], abcd[:, 1], abcd[:, 2], abcd[:, 3]
    u1 = a + b
    u2x, u2y = a - c, b + d
    u3 = c - d
    p1 = u1 ** 2 + 1
    p2 = u2x * u2y - u2x + u2y
    p3 = u3 ** 3
    return p1 + 2 * p2 - p3


def compile_subgraphs() -> list[DirectModule]:
    models = []
    for source, inputs, label in PROGRAMS:
        graph = compile_scheme(source, inputs=inputs)
        model = DirectModule(graph)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        models.append(model)
        print(f"  Compiled '{label}': {len(graph.nodes)} nodes, depth={graph.depth()}")
    return models


class HybridInterfacingModel(nn.Module):
    def __init__(self, subgraphs: list[DirectModule], n_inputs: int = 4):
        super().__init__()
        self.subgraphs = nn.ModuleList(subgraphs)
        self.proj1 = nn.Linear(n_inputs, 1, bias=False)
        self.proj2 = nn.Linear(n_inputs, 2, bias=False)
        self.proj3 = nn.Linear(n_inputs, 1, bias=False)
        self.output_weights = nn.Parameter(torch.ones(3))

    def forward(self, abcd: torch.Tensor):
        x1 = self.proj1(abcd).squeeze(1)
        x2 = self.proj2(abcd)
        x3 = self.proj3(abcd).squeeze(1)

        o1 = self.subgraphs[0].forward_batch({"x": x1})
        o2 = self.subgraphs[1].forward_batch({"x": x2[:, 0], "y": x2[:, 1]})
        o3 = self.subgraphs[2].forward_batch({"x": x3})

        outputs = torch.stack([o1, o2, o3], dim=1)
        result = (self.output_weights * outputs).sum(dim=1)
        return result

    def learned_projections(self):
        return {
            "p1 (x)": self.proj1.weight.data[0].tolist(),
            "p2 (x)": self.proj2.weight.data[0].tolist(),
            "p2 (y)": self.proj2.weight.data[1].tolist(),
            "p3 (x)": self.proj3.weight.data[0].tolist(),
            "output_w": self.output_weights.data.tolist(),
        }


class PureMLPBaseline(nn.Module):
    def __init__(self, n_inputs: int = 4, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, abcd: torch.Tensor):
        return self.net(abcd).squeeze(1)


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


def train(model, is_hybrid, epochs, batch_size, lr, x_range=2.0):
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    losses = []
    for epoch in range(epochs):
        abcd = torch.FloatTensor(batch_size, 4).uniform_(-x_range, x_range)
        y = target_function(abcd)

        y_pred = model(abcd)
        loss = F.mse_loss(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if epoch % 1000 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch:>5d}: loss = {loss.item():.4f}")
    return losses


def evaluate(model, x_range, n=50000):
    abcd = torch.FloatTensor(n, 4).uniform_(-x_range, x_range)
    y = target_function(abcd)
    with torch.no_grad():
        y_pred = model(abcd)
    return F.mse_loss(y_pred, y).item()


def visualize(hybrid, mlp, h_losses, m_losses, save_path):
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

    # --- Panel 1: Loss curves ---
    ax = fig.add_subplot(gs[0, :])
    window = 50
    h_smooth = np.convolve(h_losses, np.ones(window) / window, mode="valid")
    m_smooth = np.convolve(m_losses, np.ones(window) / window, mode="valid")
    ax.semilogy(h_smooth, "r-", linewidth=2, label="Hybrid (compiled + learned projections)")
    ax.semilogy(m_smooth, "b-", linewidth=2, alpha=0.7, label="Pure MLP")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (log)")
    ax.set_title("Training Loss")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Learned projections ---
    ax = fig.add_subplot(gs[1, 0])
    proj = hybrid.learned_projections()
    labels = ["a", "b", "c", "d"]
    true_proj = {
        "p1 (x)": [1, 1, 0, 0],
        "p2 (x)": [1, 0, -1, 0],
        "p2 (y)": [0, 1, 0, 1],
        "p3 (x)": [0, 0, 1, -1],
    }
    proj_names = list(true_proj.keys())
    x_pos = np.arange(len(labels))
    width = 0.18
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63"]
    for i, name in enumerate(proj_names):
        learned = proj[name]
        true = true_proj[name]
        offset = (i - 1.5) * width
        bars = ax.bar(x_pos + offset, learned, width * 0.85, label=f"Learned {name}",
                       color=colors[i], alpha=0.8)
        for j, (l, t) in enumerate(zip(learned, true)):
            ax.plot(x_pos[j] + offset, t, "k*", markersize=10)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Weight")
    ax.set_title("Learned Input Projections (stars = true)")
    ax.legend(fontsize=8, loc="upper right")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    # --- Panel 3: Learned output weights ---
    ax = fig.add_subplot(gs[1, 1])
    out_w = proj["output_w"]
    true_w = TRUE_OUTPUT_WEIGHTS
    sg_labels = [PROGRAMS[i][2] for i in range(3)]
    x_pos = np.arange(3)
    ax.bar(x_pos - 0.15, out_w, 0.3, label="Learned", color="#2196F3", alpha=0.8)
    ax.bar(x_pos + 0.15, true_w, 0.3, label="True", color="#FF9800", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sg_labels, fontsize=10)
    ax.set_ylabel("Weight")
    ax.set_title("Learned Output Combination Weights")
    ax.legend(fontsize=11)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    # --- Panel 4: 1D slice comparison (vary a, fix b=c=d=1) ---
    ax = fig.add_subplot(gs[2, 0])
    a_range = torch.linspace(-4, 4, 500)
    abcd_slice = torch.stack([a_range, torch.ones(500), torch.ones(500), torch.ones(500)], dim=1)
    y_true = target_function(abcd_slice).numpy()
    with torch.no_grad():
        y_hyb = hybrid(abcd_slice).numpy()
        y_mlp = mlp(abcd_slice).numpy()
    ax.plot(a_range.numpy(), y_true, "k-", linewidth=2.5, label="True f(a,1,1,1)")
    ax.plot(a_range.numpy(), y_hyb, "r--", linewidth=2, label="Hybrid")
    ax.plot(a_range.numpy(), y_mlp, "b--", linewidth=2, alpha=0.7, label="Pure MLP")
    ax.axvline(-2, color="gray", linestyle=":", alpha=0.3)
    ax.axvline(2, color="gray", linestyle=":", alpha=0.3)
    ax.fill_betweenx([-100, 200], -2, 2, alpha=0.05, color="green", label="Training range")
    ax.set_xlim(-4, 4)
    ax.set_ylim(min(y_true.min(), y_mlp.min()) - 5, max(y_true.max(), y_mlp.max()) + 5)
    ax.set_xlabel("a (b=c=d=1)")
    ax.set_ylabel("f(a, 1, 1, 1)")
    ax.set_title("1D Slice: Extrapolation along a")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel 5: 1D slice (vary c, fix a=b=d=1) ---
    ax = fig.add_subplot(gs[2, 1])
    c_range = torch.linspace(-4, 4, 500)
    abcd_slice = torch.stack([torch.ones(500), torch.ones(500), c_range, torch.ones(500)], dim=1)
    y_true = target_function(abcd_slice).numpy()
    with torch.no_grad():
        y_hyb = hybrid(abcd_slice).numpy()
        y_mlp = mlp(abcd_slice).numpy()
    ax.plot(c_range.numpy(), y_true, "k-", linewidth=2.5, label="True f(1,1,c,1)")
    ax.plot(c_range.numpy(), y_hyb, "r--", linewidth=2, label="Hybrid")
    ax.plot(c_range.numpy(), y_mlp, "b--", linewidth=2, alpha=0.7, label="Pure MLP")
    ax.axvline(-2, color="gray", linestyle=":", alpha=0.3)
    ax.axvline(2, color="gray", linestyle=":", alpha=0.3)
    ax.fill_betweenx([-100, 200], -2, 2, alpha=0.05, color="green", label="Training range")
    ax.set_xlim(-4, 4)
    ax.set_ylim(min(y_true.min(), y_mlp.min()) - 5, max(y_true.max(), y_mlp.max()) + 5)
    ax.set_xlabel("c (a=b=d=1)")
    ax.set_ylabel("f(1, 1, c, 1)")
    ax.set_title("1D Slice: Extrapolation along c (cubic via p3)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Neural Compiler: Learned Input Interfacing with Compiled Subgraphs",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Hybrid interfacing example")
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--x-range", type=float, default=2.0)
    parser.add_argument("--save-fig", default="examples/hybrid_interfacing.png")
    args = parser.parse_args()

    print("Compiling Scheme programs to frozen compiled modules...")
    subgraphs = compile_subgraphs()

    hybrid = HybridInterfacingModel(subgraphs)
    mlp = PureMLPBaseline(hidden=64)
    h_params = count_params(hybrid)
    m_params = count_params(mlp)
    frozen = count_frozen_structure(subgraphs)
    print(f"\nHybrid model trainable params: {h_params}")
    print(f"  Frozen subgraph structure: {frozen['nodes']} nodes, {frozen['edges']} edges, {frozen['consts']} const floats")
    print(f"Pure MLP trainable params:     {m_params}")

    print(f"\nTraining hybrid model ({args.epochs} epochs, x in [-{args.x_range}, {args.x_range}])...")
    h_losses = train(hybrid, is_hybrid=True, epochs=args.epochs,
                     batch_size=args.batch_size, lr=args.lr, x_range=args.x_range)

    print(f"\nTraining pure MLP ({args.epochs} epochs)...")
    m_losses = train(mlp, is_hybrid=False, epochs=args.epochs,
                     batch_size=args.batch_size, lr=args.lr, x_range=args.x_range)

    print("\nLearned projections (true values in brackets):")
    proj = hybrid.learned_projections()
    true_labels = {
        "p1 (x)": "[1, 1, 0, 0] = a+b",
        "p2 (x)": "[1, 0, -1, 0] = a-c",
        "p2 (y)": "[0, 1, 0, 1] = b+d",
        "p3 (x)": "[0, 0, 1, -1] = c-d",
        "output_w": "[1, 2, -1]",
    }
    for name, weights in proj.items():
        w_str = "[" + ", ".join(f"{w:6.3f}" for w in weights) + "]"
        print(f"  {name:12s}: {w_str}  (true: {true_labels[name]})")

    print(f"\nTest MSE (in-distribution, x in [-{args.x_range}, {args.x_range}]):")
    h_mse = evaluate(hybrid, args.x_range)
    m_mse = evaluate(mlp, args.x_range)
    print(f"  Hybrid: {h_mse:.6f}")
    print(f"  MLP:    {m_mse:.6f}")
    print(f"  Ratio:  MLP is {m_mse / max(h_mse, 1e-12):.1f}x worse")

    extrap = args.x_range * 2
    print(f"\nExtrapolation MSE (x in [-{extrap}, {extrap}]):")
    h_mse_e = evaluate(hybrid, extrap)
    m_mse_e = evaluate(mlp, extrap)
    print(f"  Hybrid: {h_mse_e:.6f}")
    print(f"  MLP:    {m_mse_e:.6f}")
    print(f"  Ratio:  MLP is {m_mse_e / max(h_mse_e, 1e-12):.1f}x worse")

    visualize(hybrid, mlp, h_losses, m_losses, args.save_fig)


if __name__ == "__main__":
    main()

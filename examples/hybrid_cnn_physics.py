############################################################
#
# Neural Compiler: Compiling scheme into composable and
#                  differentiable neural network representations
#
# hybrid_cnn_physics.py: CNN extracting physical quantities from images feeding compiled physics modules
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Hybrid CNN + compiled subgraph architecture for visual physics.

Demonstrates that a CNN can extract physical quantities from synthetic images
and feed them to compiled modules that compute exact physics. The CNN
handles perception; the compiled subgraphs handle computation.

Target: Total mechanical energy E = ½mv² + mgh
where (m, v, h) are encoded as bar heights in 32x32 synthetic images.

Compiled subgraphs:
    1. kinetic(m, v) = ½mv²   — Scheme: (* 0.5 (* m (* v v)))
    2. potential(m, h) = mgh   — Scheme: (* 9.81 (* m h))

Architecture:
    Image → CNN → (m̂, v̂, ĥ) → compiled KE, PE → Linear(2→1) → Ê

Both models (hybrid and baseline) share the same CNN backbone structure and
receive identical auxiliary extraction supervision — the only difference is
compiled physics (hybrid) vs learned MLP head (baseline).

Gradient flows: loss → combine → through frozen subgraphs → CNN backbone.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_compiler.compiler import compile_scheme
from neural_compiler.evaluator.direct_module import DirectModule

KINETIC_SRC = "(* 0.5 (* m (* v v)))"
POTENTIAL_SRC = "(* 9.81 (* m h))"

IMG_SIZE = 64
BAR_WIDTH = 8
BAR_CENTERS = [12, 32, 52]
MAX_DISPLAY_VAL = 10.0


def target_energy(m: torch.Tensor, v: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    return 0.5 * m * v ** 2 + 9.81 * m * h


def compile_subgraphs():
    graph_ke = compile_scheme(KINETIC_SRC, inputs={"m": None, "v": None})
    model_ke = DirectModule(graph_ke)
    for p in model_ke.parameters():
        p.requires_grad = False
    model_ke.eval()
    print(f"  Compiled kinetic energy: {len(graph_ke.nodes)} nodes")

    graph_pe = compile_scheme(POTENTIAL_SRC, inputs={"m": None, "h": None})
    model_pe = DirectModule(graph_pe)
    for p in model_pe.parameters():
        p.requires_grad = False
    model_pe.eval()
    print(f"  Compiled potential energy: {len(graph_pe.nodes)} nodes")

    return model_ke, model_pe


def generate_batch(batch_size, val_range=(1.0, 5.0), noise_std=0.1):
    m = torch.FloatTensor(batch_size).uniform_(*val_range)
    v = torch.FloatTensor(batch_size).uniform_(*val_range)
    h = torch.FloatTensor(batch_size).uniform_(*val_range)

    images = torch.zeros(batch_size, 1, IMG_SIZE, IMG_SIZE)
    row_idx = torch.arange(IMG_SIZE).unsqueeze(0)

    for vals, cx in zip([m, v, h], BAR_CENTERS):
        bar_px = (vals / MAX_DISPLAY_VAL * (IMG_SIZE - 2)).long().clamp(min=1)
        thresholds = (IMG_SIZE - bar_px).unsqueeze(1)
        mask = (row_idx >= thresholds).float()
        x_start = cx - BAR_WIDTH // 2
        x_end = cx + BAR_WIDTH // 2
        images[:, 0, :, x_start:x_end] = mask.unsqueeze(2).expand(
            -1, -1, BAR_WIDTH
        )

    images += torch.randn_like(images) * noise_std
    images.clamp_(0, 1)

    energy = target_energy(m, v, h)
    return images, m, v, h, energy


class HybridCNNPhysics(nn.Module):
    def __init__(self, sg_kinetic: DirectModule, sg_potential: DirectModule):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(512, 64),
            nn.ReLU(),
        )
        self.extract = nn.Linear(64, 3)
        self.sg_kinetic = sg_kinetic
        self.sg_potential = sg_potential
        self.combine = nn.Linear(2, 1)
        nn.init.ones_(self.combine.weight)
        nn.init.zeros_(self.combine.bias)

    def forward(self, images: torch.Tensor):
        feat = self.backbone(images)
        raw = self.extract(feat)
        m_hat = F.softplus(raw[:, 0])
        v_hat = F.softplus(raw[:, 1])
        h_hat = F.softplus(raw[:, 2])

        ke = self.sg_kinetic.forward_batch({"m": m_hat, "v": v_hat})
        pe = self.sg_potential.forward_batch({"m": m_hat, "h": h_hat})

        out = self.combine(torch.stack([ke, pe], dim=1)).squeeze(1)
        return out, m_hat, v_hat, h_hat

    def extract_quantities(self, images: torch.Tensor):
        with torch.no_grad():
            feat = self.backbone(images)
            raw = self.extract(feat)
            return F.softplus(raw[:, 0]), F.softplus(raw[:, 1]), F.softplus(raw[:, 2])


class PureCNNBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(512, 64),
            nn.ReLU(),
        )
        self.extract = nn.Linear(64, 3)
        self.head = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, images: torch.Tensor):
        feat = self.backbone(images)
        raw = self.extract(feat)
        quantities = F.softplus(raw)
        out = self.head(quantities).squeeze(1)
        return out, quantities[:, 0], quantities[:, 1], quantities[:, 2]


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
            s.edge_index.shape[1]
            for s in sg._data_template.edge_stores
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


def train(model, epochs, batch_size, lr, val_range=(1.0, 5.0), extract_weight=1.0,
          noise_std=0.05):
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    losses = []
    for epoch in range(epochs):
        images, m, v, h, energy = generate_batch(batch_size, val_range,
                                                  noise_std=noise_std)
        pred, m_hat, v_hat, h_hat = model(images)
        energy_loss = F.mse_loss(pred, energy)
        extract_loss = (
            F.mse_loss(m_hat, m) + F.mse_loss(v_hat, v) + F.mse_loss(h_hat, h)
        ) / 3
        loss = energy_loss + extract_weight * extract_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(energy_loss.item())
        if epoch % 500 == 0 or epoch == epochs - 1:
            print(
                f"  Epoch {epoch:>5d}: energy_loss = {energy_loss.item():.4f}"
                f"  extract_loss = {extract_loss.item():.4f}"
            )
    return losses


def evaluate(model, val_range, n_batches=10, batch_size=500, noise_std=0.05):
    total_mse = 0.0
    for _ in range(n_batches):
        images, m, v, h, energy = generate_batch(batch_size, val_range,
                                                  noise_std=noise_std)
        with torch.no_grad():
            pred, _, _, _ = model(images)
        total_mse += F.mse_loss(pred, energy).item()
    return total_mse / n_batches


def visualize(hybrid, baseline, h_losses, b_losses, save_path, noise_std=0.05):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Loss curves
    ax = axes[0, 0]
    window = 30
    h_smooth = np.convolve(h_losses, np.ones(window) / window, mode="valid")
    b_smooth = np.convolve(b_losses, np.ones(window) / window, mode="valid")
    ax.semilogy(h_smooth, "r-", linewidth=2, label="Hybrid (CNN + compiled physics)")
    ax.semilogy(b_smooth, "b-", linewidth=2, alpha=0.7, label="Pure CNN baseline")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Energy MSE (log)")
    ax.set_title("Training Loss (energy prediction only)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Extracted quantities (hybrid model)
    ax = axes[0, 1]
    images, m, v, h, energy = generate_batch(500, val_range=(1.0, 5.0),
                                              noise_std=noise_std)
    m_hat, v_hat, h_hat = hybrid.extract_quantities(images)
    for true_v, pred_v, label, color in [
        (m, m_hat, "mass (m)", "#2196F3"),
        (v, v_hat, "velocity (v)", "#FF9800"),
        (h, h_hat, "height (h)", "#4CAF50"),
    ]:
        ax.scatter(
            true_v.numpy(), pred_v.numpy(), alpha=0.3, s=8, color=color, label=label
        )
    lims = [0, 7]
    ax.plot(lims, lims, "k--", linewidth=1, alpha=0.5, label="perfect")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("True value")
    ax.set_ylabel("Extracted value")
    ax.set_title("CNN-Extracted Physical Quantities")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    # Panel 3: In-distribution prediction
    ax = axes[1, 0]
    images, m, v, h, energy = generate_batch(500, val_range=(1.0, 5.0),
                                              noise_std=noise_std)
    with torch.no_grad():
        h_pred, _, _, _ = hybrid(images)
        b_pred, _, _, _ = baseline(images)
    h_pred = h_pred.numpy()
    b_pred = b_pred.numpy()
    e_np = energy.numpy()
    ax.scatter(e_np, h_pred, alpha=0.3, s=8, color="red", label="Hybrid")
    ax.scatter(e_np, b_pred, alpha=0.3, s=8, color="blue", label="Baseline")
    pad = (e_np.max() - e_np.min()) * 0.05
    lims = [e_np.min() - pad, e_np.max() + pad]
    ax.plot(lims, lims, "k--", linewidth=1, alpha=0.5)
    ax.set_xlabel("True energy")
    ax.set_ylabel("Predicted energy")
    ax.set_title("In-Distribution [1, 5]")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 4: Extrapolation prediction
    ax = axes[1, 1]
    images, m, v, h, energy = generate_batch(500, val_range=(5.0, 8.0),
                                              noise_std=noise_std)
    with torch.no_grad():
        h_pred, _, _, _ = hybrid(images)
        b_pred, _, _, _ = baseline(images)
    h_pred = h_pred.numpy()
    b_pred = b_pred.numpy()
    e_np = energy.numpy()
    ax.scatter(e_np, h_pred, alpha=0.3, s=8, color="red", label="Hybrid")
    ax.scatter(e_np, b_pred, alpha=0.3, s=8, color="blue", label="Baseline")
    pad = (e_np.max() - e_np.min()) * 0.05
    lims = [e_np.min() - pad, e_np.max() + pad]
    ax.plot(lims, lims, "k--", linewidth=1, alpha=0.5)
    ax.set_xlabel("True energy")
    ax.set_ylabel("Predicted energy")
    ax.set_title("Extrapolation [5, 8] (unseen range)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Neural Compiler: CNN + Compiled Physics Subgraphs",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--extract-weight", type=float, default=1.0)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--save-fig", default="examples/hybrid_cnn_physics.png")
    args = parser.parse_args()

    print("Compiling Scheme programs to frozen compiled modules...")
    sg_ke, sg_pe = compile_subgraphs()

    hybrid = HybridCNNPhysics(sg_ke, sg_pe)
    baseline = PureCNNBaseline()
    frozen = count_frozen_structure([sg_ke, sg_pe])
    h_params = count_params(hybrid)
    b_params = count_params(baseline)

    h_cnn = sum(p.numel() for p in hybrid.backbone.parameters()) + sum(
        p.numel() for p in hybrid.extract.parameters()
    )
    h_head = sum(p.numel() for p in hybrid.combine.parameters())

    print(f"\nHybrid trainable params: {h_params}")
    print(f"  CNN backbone + extract: {h_cnn}")
    print(f"  Physics combination: {h_head}")
    print(
        f"  Frozen subgraph structure: {frozen['nodes']} nodes, "
        f"{frozen['edges']} edges, {frozen['consts']} const floats"
    )
    print(f"Pure CNN baseline trainable params: {b_params}")
    print(f"\nBoth models receive identical auxiliary extraction supervision.")
    print(f"Image noise: {args.noise_std}, extraction weight: {args.extract_weight}")

    print(f"\nTraining hybrid model ({args.epochs} epochs)...")
    h_losses = train(
        hybrid, args.epochs, args.batch_size, args.lr,
        extract_weight=args.extract_weight, noise_std=args.noise_std,
    )

    print(f"\nTraining pure CNN baseline ({args.epochs} epochs)...")
    b_losses = train(
        baseline, args.epochs, args.batch_size, args.lr,
        extract_weight=args.extract_weight, noise_std=args.noise_std,
    )

    cw = hybrid.combine.weight.data[0].tolist()
    cb = hybrid.combine.bias.data[0].item()
    print(f"\nLearned combination weights:")
    print(f"  w(KE)={cw[0]:.4f}  w(PE)={cw[1]:.4f}  bias={cb:.4f}")
    print(f"  True: w(KE)=1.0  w(PE)=1.0  bias=0.0")

    h_mse = evaluate(hybrid, (1.0, 5.0), noise_std=args.noise_std)
    b_mse = evaluate(baseline, (1.0, 5.0), noise_std=args.noise_std)
    print(f"\nTest MSE (in-distribution [1, 5]):")
    print(f"  Hybrid:   {h_mse:.4f}")
    print(f"  Baseline: {b_mse:.4f}")
    ratio_in = b_mse / max(h_mse, 1e-12)
    print(f"  Ratio:    Baseline is {ratio_in:.1f}x worse")

    h_ext = evaluate(hybrid, (5.0, 8.0), noise_std=args.noise_std)
    b_ext = evaluate(baseline, (5.0, 8.0), noise_std=args.noise_std)
    print(f"\nTest MSE (extrapolation [5, 8]):")
    print(f"  Hybrid:   {h_ext:.4f}")
    print(f"  Baseline: {b_ext:.4f}")
    ratio_ext = b_ext / max(h_ext, 1e-12)
    print(f"  Ratio:    Baseline is {ratio_ext:.1f}x worse")

    visualize(hybrid, baseline, h_losses, b_losses, args.save_fig,
              noise_std=args.noise_std)


if __name__ == "__main__":
    main()

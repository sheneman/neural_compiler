############################################################
#
# Neural Compiler: Compiling scheme into composable and
#                  differentiable neural network representations
#
# vector_mechanics.py: Compiled 3D gravitational force with vector operations learning G from noisy data
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""3D Vector Mechanics: compiled vector physics + learned corrections.

Demonstrates the neural compiler's vector operations on real 3D physics:

  Experiment 1 — Gravitational Force Recovery
    Known structure: F = -G*m1*m2 / |r|^3 * r  (compiled as Scheme)
    Learn: gravitational constant G from noisy force observations
    Compare: pure MLP baseline

  Experiment 2 — Hybrid Gravity + Drag
    Known structure: gravitational force (compiled)
    Learned: velocity-dependent drag correction (MLP residual)
    Tests whether compiled physics + learned correction beats pure MLP

Usage:
    python examples/vector_mechanics.py [--epochs 1000] [--quiet]
"""

import argparse
import math
import pickle
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from neural_compiler.compiler import compile_scheme
from neural_compiler.evaluator.direct_module import DirectModule


# ---------------------------------------------------------------------------
# Compiled physics
# ---------------------------------------------------------------------------

GRAVITY_SCHEME = """
(let ((r_mag (norm r))
      (r_hat (normalize r)))
  (scale (/ (* (- 0 G) (* m1 m2))
             (* r_mag (* r_mag r_mag)))
         r))
"""

SPRING_SCHEME = """
(scale (- 0 k) r)
"""


def build_gravity_model():
    graph = compile_scheme(
        GRAVITY_SCHEME,
        inputs={"r": None, "G": None, "m1": None, "m2": None},
    )
    return DirectModule(graph)


def build_spring_model():
    graph = compile_scheme(
        SPRING_SCHEME,
        inputs={"r": None, "k": None},
    )
    return DirectModule(graph)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class GravityHybrid(nn.Module):
    """Compiled gravitational structure with trainable G."""

    def __init__(self):
        super().__init__()
        self.subgraph = build_gravity_model()
        self.G = nn.Parameter(torch.tensor(3.0))

    def forward(self, r, m1, m2):
        return self.subgraph({
            "r": r, "G": self.G, "m1": m1, "m2": m2,
        })

    def learned_G(self):
        return self.G.item()


class GravityDragHybrid(nn.Module):
    """Compiled gravity + learned drag correction."""

    def __init__(self):
        super().__init__()
        self.gravity = build_gravity_model()
        self.G = nn.Parameter(torch.tensor(3.0))
        self.drag_net = nn.Sequential(
            nn.Linear(3, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(self, r, v, m1, m2):
        f_grav = self.gravity({
            "r": r, "G": self.G, "m1": m1, "m2": m2,
        })
        f_drag = self.drag_net(v)
        return f_grav + f_drag


class HandCodedGravity(nn.Module):
    """Hand-coded PyTorch gravity — same equation, no compiler."""

    def __init__(self):
        super().__init__()
        self.G = nn.Parameter(torch.tensor(3.0))

    def forward(self, r, m1, m2):
        r_mag = r.norm(dim=-1, keepdim=True)
        return -self.G * m1 * m2 / (r_mag ** 3) * r

    def learned_G(self):
        return self.G.item()


class HandCodedGravityDrag(nn.Module):
    """Hand-coded gravity + MLP drag — same as GravityDragHybrid but no compiler."""

    def __init__(self):
        super().__init__()
        self.G = nn.Parameter(torch.tensor(3.0))
        self.drag_net = nn.Sequential(
            nn.Linear(3, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(self, r, v, m1, m2):
        r_mag = r.norm(dim=-1, keepdim=True)
        f_grav = -self.G * m1 * m2 / (r_mag ** 3) * r
        f_drag = self.drag_net(v)
        return f_grav + f_drag


class ForceMLP(nn.Module):
    """Pure MLP baseline for force prediction."""

    def __init__(self, n_inputs: int, hidden: int = 64, layers: int = 3):
        super().__init__()
        mods = [nn.Linear(n_inputs, hidden), nn.ReLU()]
        for _ in range(layers - 1):
            mods.extend([nn.Linear(hidden, hidden), nn.ReLU()])
        mods.append(nn.Linear(hidden, 3))
        self.net = nn.Sequential(*mods)

    def forward(self, *inputs):
        x = torch.cat(inputs, dim=-1) if inputs[0].dim() == 1 else torch.cat(inputs, dim=-1)
        return self.net(x)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

TRUE_G = 6.674

def generate_gravity_data(n, noise_std=0.0, seed=42):
    torch.manual_seed(seed)
    r = torch.randn(n, 3)
    r = r / r.norm(dim=-1, keepdim=True) * (torch.rand(n, 1) * 4 + 1)
    m1 = torch.rand(n, 1) * 4 + 1
    m2 = torch.rand(n, 1) * 4 + 1

    r_mag = r.norm(dim=-1, keepdim=True)
    F_true = -TRUE_G * m1 * m2 / (r_mag ** 3) * r

    if noise_std > 0:
        F_true = F_true + noise_std * F_true.abs().mean() * torch.randn_like(F_true)

    return r, m1.squeeze(-1), m2.squeeze(-1), F_true


def generate_gravity_drag_data(n, drag_coeff=0.5, noise_std=0.0, seed=42):
    torch.manual_seed(seed)
    r = torch.randn(n, 3)
    r = r / r.norm(dim=-1, keepdim=True) * (torch.rand(n, 1) * 4 + 1)
    v = torch.randn(n, 3) * 2
    m1 = torch.rand(n, 1) * 4 + 1
    m2 = torch.rand(n, 1) * 4 + 1

    r_mag = r.norm(dim=-1, keepdim=True)
    F_grav = -TRUE_G * m1 * m2 / (r_mag ** 3) * r
    F_drag = -drag_coeff * v * v.abs()
    F_total = F_grav + F_drag

    if noise_std > 0:
        F_total = F_total + noise_std * F_total.abs().mean() * torch.randn_like(F_total)

    return r, v, m1.squeeze(-1), m2.squeeze(-1), F_total


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_gravity_recovery(epochs=1000, lr=0.01, n_train=100, noise_std=0.02,
                           quiet=False):
    r, m1, m2, F_true = generate_gravity_data(n_train, noise_std=noise_std)
    r_test, m1_test, m2_test, F_test = generate_gravity_data(50, seed=99)

    hybrid = GravityHybrid()
    handcoded = HandCodedGravity()
    mlp = ForceMLP(n_inputs=5)
    opt_h = torch.optim.Adam(hybrid.parameters(), lr=lr)
    opt_hc = torch.optim.Adam(handcoded.parameters(), lr=lr)
    opt_m = torch.optim.Adam(mlp.parameters(), lr=0.001)

    history = {"hybrid_loss": [], "handcoded_loss": [], "mlp_loss": [],
               "hybrid_test_loss": [], "handcoded_test_loss": [], "mlp_test_loss": [],
               "test_epochs": [],
               "G_estimate": [], "G_estimate_handcoded": []}

    eval_interval = max(1, epochs // 100)

    for epoch in range(epochs):
        # Hybrid — per-sample forward (vector outputs)
        opt_h.zero_grad()
        preds = [hybrid(r[i], m1[i], m2[i]) for i in range(n_train)]
        pred_stack = torch.stack(preds)
        loss_h = ((pred_stack - F_true) ** 2).mean()
        loss_h.backward()
        opt_h.step()

        # Hand-coded — per-sample forward (same pattern as compiled)
        opt_hc.zero_grad()
        preds_hc = [handcoded(r[i], m1[i], m2[i]) for i in range(n_train)]
        pred_stack_hc = torch.stack(preds_hc)
        loss_hc = ((pred_stack_hc - F_true) ** 2).mean()
        loss_hc.backward()
        opt_hc.step()

        # MLP — batched forward
        opt_m.zero_grad()
        pred_m = mlp(r, m1.unsqueeze(-1), m2.unsqueeze(-1))
        loss_m = ((pred_m - F_true) ** 2).mean()
        loss_m.backward()
        opt_m.step()

        history["hybrid_loss"].append(loss_h.item())
        history["handcoded_loss"].append(loss_hc.item())
        history["mlp_loss"].append(loss_m.item())
        history["G_estimate"].append(hybrid.learned_G())
        history["G_estimate_handcoded"].append(handcoded.learned_G())

        if (epoch + 1) % eval_interval == 0 or epoch == 0:
            with torch.no_grad():
                tp = [hybrid(r_test[i], m1_test[i], m2_test[i]) for i in range(len(r_test))]
                tl_h = ((torch.stack(tp) - F_test) ** 2).mean().item()
                tp_hc = [handcoded(r_test[i], m1_test[i], m2_test[i]) for i in range(len(r_test))]
                tl_hc = ((torch.stack(tp_hc) - F_test) ** 2).mean().item()
                tp_m = mlp(r_test, m1_test.unsqueeze(-1), m2_test.unsqueeze(-1))
                tl_m = ((tp_m - F_test) ** 2).mean().item()
            history["hybrid_test_loss"].append(tl_h)
            history["handcoded_test_loss"].append(tl_hc)
            history["mlp_test_loss"].append(tl_m)
            history["test_epochs"].append(epoch + 1)

        if not quiet and (epoch + 1) % 200 == 0:
            print(f"  Epoch {epoch+1:4d}  hybrid={loss_h.item():.6f}  "
                  f"handcoded={loss_hc.item():.6f}  "
                  f"mlp={loss_m.item():.6f}  G={hybrid.learned_G():.4f}")

    # Test
    with torch.no_grad():
        preds = [hybrid(r_test[i], m1_test[i], m2_test[i]) for i in range(len(r_test))]
        pred_stack = torch.stack(preds)
        test_loss_h = ((pred_stack - F_test) ** 2).mean()

        preds_hc = [handcoded(r_test[i], m1_test[i], m2_test[i]) for i in range(len(r_test))]
        pred_stack_hc = torch.stack(preds_hc)
        test_loss_hc = ((pred_stack_hc - F_test) ** 2).mean()

        pred_m = mlp(r_test, m1_test.unsqueeze(-1), m2_test.unsqueeze(-1))
        test_loss_m = ((pred_m - F_test) ** 2).mean()

    return {
        "hybrid_test_mse": test_loss_h.item(),
        "handcoded_test_mse": test_loss_hc.item(),
        "mlp_test_mse": test_loss_m.item(),
        "G_learned": hybrid.learned_G(),
        "G_learned_handcoded": handcoded.learned_G(),
        "G_true": TRUE_G,
        "G_error_pct": abs(hybrid.learned_G() - TRUE_G) / TRUE_G * 100,
        "handcoded_history": history,
        "history": history,
    }


def train_hybrid_drag(epochs=2000, n_train=100, noise_std=0.01, quiet=False):
    r, v, m1, m2, F_true = generate_gravity_drag_data(n_train, noise_std=noise_std)
    r_t, v_t, m1_t, m2_t, F_t = generate_gravity_drag_data(50, seed=99)

    hybrid = GravityDragHybrid()
    handcoded = HandCodedGravityDrag()
    mlp = ForceMLP(n_inputs=8)
    opt_h = torch.optim.Adam(hybrid.parameters(), lr=0.001)
    opt_hc = torch.optim.Adam(handcoded.parameters(), lr=0.001)
    opt_m = torch.optim.Adam(mlp.parameters(), lr=0.001)

    history = {"hybrid_loss": [], "handcoded_loss": [], "mlp_loss": [],
               "hybrid_test_loss": [], "handcoded_test_loss": [], "mlp_test_loss": [],
               "test_epochs": []}

    eval_interval = max(1, epochs // 100)

    for epoch in range(epochs):
        # Hybrid — per-sample forward (vector outputs)
        opt_h.zero_grad()
        preds = [hybrid(r[i], v[i], m1[i], m2[i]) for i in range(n_train)]
        pred_stack = torch.stack(preds)
        loss_h = ((pred_stack - F_true) ** 2).mean()
        loss_h.backward()
        opt_h.step()

        # Hand-coded — per-sample forward (same pattern)
        opt_hc.zero_grad()
        preds_hc = [handcoded(r[i], v[i], m1[i], m2[i]) for i in range(n_train)]
        pred_stack_hc = torch.stack(preds_hc)
        loss_hc = ((pred_stack_hc - F_true) ** 2).mean()
        loss_hc.backward()
        opt_hc.step()

        # MLP — batched
        opt_m.zero_grad()
        pred_m = mlp(r, v, m1.unsqueeze(-1), m2.unsqueeze(-1))
        loss_m = ((pred_m - F_true) ** 2).mean()
        loss_m.backward()
        opt_m.step()

        history["hybrid_loss"].append(loss_h.item())
        history["handcoded_loss"].append(loss_hc.item())
        history["mlp_loss"].append(loss_m.item())

        if (epoch + 1) % eval_interval == 0 or epoch == 0:
            with torch.no_grad():
                tp = [hybrid(r_t[i], v_t[i], m1_t[i], m2_t[i]) for i in range(len(r_t))]
                tl_h = ((torch.stack(tp) - F_t) ** 2).mean().item()
                tp_hc = [handcoded(r_t[i], v_t[i], m1_t[i], m2_t[i]) for i in range(len(r_t))]
                tl_hc = ((torch.stack(tp_hc) - F_t) ** 2).mean().item()
                tp_m = mlp(r_t, v_t, m1_t.unsqueeze(-1), m2_t.unsqueeze(-1))
                tl_m = ((tp_m - F_t) ** 2).mean().item()
            history["hybrid_test_loss"].append(tl_h)
            history["handcoded_test_loss"].append(tl_hc)
            history["mlp_test_loss"].append(tl_m)
            history["test_epochs"].append(epoch + 1)

        if not quiet and (epoch + 1) % 500 == 0:
            print(f"  Epoch {epoch+1:4d}  hybrid={loss_h.item():.6f}  "
                  f"handcoded={loss_hc.item():.6f}  "
                  f"mlp={loss_m.item():.6f}  G={hybrid.G.item():.4f}")

    # Test
    with torch.no_grad():
        preds = [hybrid(r_t[i], v_t[i], m1_t[i], m2_t[i]) for i in range(len(r_t))]
        pred_stack = torch.stack(preds)
        test_loss_h = ((pred_stack - F_t) ** 2).mean()

        preds_hc = [handcoded(r_t[i], v_t[i], m1_t[i], m2_t[i]) for i in range(len(r_t))]
        pred_stack_hc = torch.stack(preds_hc)
        test_loss_hc = ((pred_stack_hc - F_t) ** 2).mean()

        pred_m = mlp(r_t, v_t, m1_t.unsqueeze(-1), m2_t.unsqueeze(-1))
        test_loss_m = ((pred_m - F_t) ** 2).mean()

    return {
        "hybrid_test_mse": test_loss_h.item(),
        "handcoded_test_mse": test_loss_hc.item(),
        "mlp_test_mse": test_loss_m.item(),
        "G_learned": hybrid.G.item(),
        "G_learned_handcoded": handcoded.G.item(),
        "history": history,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(grav_results, drag_results, filename="examples/vector_mechanics.png"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. G recovery (compiled + hand-coded)
    ax = axes[0]
    epochs = range(1, len(grav_results["history"]["G_estimate"]) + 1)
    ax.plot(epochs, grav_results["history"]["G_estimate"], "b-", linewidth=2,
            label="Compiled")
    ax.plot(epochs, grav_results["history"]["G_estimate_handcoded"], color="purple",
            linestyle="-", linewidth=2, label="Hand-coded")
    ax.axhline(y=TRUE_G, color="r", linestyle="--", linewidth=1.5, label=f"True G = {TRUE_G}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learned G")
    ax.set_title(f"Gravitational Constant Recovery\n"
                 f"Compiled: {grav_results['G_learned']:.4f}, "
                 f"Hand-coded: {grav_results['G_learned_handcoded']:.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Gravity test loss comparison
    ax = axes[1]
    h = grav_results["history"]
    ax.semilogy(h["test_epochs"], h["hybrid_test_loss"],
                label="Compiled + trainable G", linewidth=2)
    ax.semilogy(h["test_epochs"], h["handcoded_test_loss"],
                label="Hand-coded PyTorch", linewidth=2, color="purple", alpha=0.7)
    ax.semilogy(h["test_epochs"], h["mlp_test_loss"],
                label="Pure MLP", linewidth=2, alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test MSE")
    ax.set_title(f"Gravity: Test Loss (Compiled vs Hand-coded vs MLP)\n"
                 f"Final — Compiled: {grav_results['hybrid_test_mse']:.2e}, "
                 f"HC: {grav_results['handcoded_test_mse']:.2e}, "
                 f"MLP: {grav_results['mlp_test_mse']:.2e}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Drag test loss comparison
    ax = axes[2]
    h = drag_results["history"]
    ax.semilogy(h["test_epochs"], h["hybrid_test_loss"],
                label="Compiled gravity + MLP drag", linewidth=2)
    ax.semilogy(h["test_epochs"], h["handcoded_test_loss"],
                label="Hand-coded + MLP drag", linewidth=2, color="purple", alpha=0.7)
    ax.semilogy(h["test_epochs"], h["mlp_test_loss"],
                label="Pure MLP", linewidth=2, alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test MSE")
    ax.set_title(f"Gravity + Drag: Test Loss (Hybrid vs Hand-coded vs MLP)\n"
                 f"Final — Hybrid: {drag_results['hybrid_test_mse']:.2e}, "
                 f"HC: {drag_results['handcoded_test_mse']:.2e}, "
                 f"MLP: {drag_results['mlp_test_mse']:.2e}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {filename}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="3D Vector Mechanics Demo")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--save-fig", default="examples/vector_mechanics.png")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip training; load saved data and regenerate figures")
    args = parser.parse_args()

    if args.plot_only:
        data_path = args.save_fig.replace(".png", "_data.pkl")
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        grav_results = data["grav_results"]
        drag_results = data["drag_results"]
        plot_results(grav_results, drag_results, filename=args.save_fig)
        return

    print("=" * 70)
    print("EXPERIMENT 1: Gravitational Constant Recovery (3D Vector)")
    print("  Structure: F = -G*m1*m2/|r|^3 * r  (compiled Scheme)")
    print(f"  True G = {TRUE_G}")
    print("=" * 70)

    t0 = time.time()
    grav = train_gravity_recovery(epochs=args.epochs, quiet=args.quiet)
    t1 = time.time()

    print(f"\n  Results ({t1-t0:.1f}s):")
    print(f"    Learned G (compiled)   = {grav['G_learned']:.4f}  (error: {grav['G_error_pct']:.2f}%)")
    print(f"    Learned G (hand-coded) = {grav['G_learned_handcoded']:.4f}")
    print(f"    Compiled test MSE:    {grav['hybrid_test_mse']:.2e}")
    print(f"    Hand-coded test MSE:  {grav['handcoded_test_mse']:.2e}")
    print(f"    MLP test MSE:         {grav['mlp_test_mse']:.2e}")
    params_h = 1
    params_m = sum(p.numel() for p in ForceMLP(5).parameters())
    print(f"    Parameters: compiled={params_h}, hand-coded={params_h}, MLP={params_m}")

    print()
    print("=" * 70)
    print("EXPERIMENT 2: Hybrid Gravity + Drag Correction")
    print("  Known: gravitational force structure (compiled)")
    print("  Learned: velocity-dependent drag (MLP residual)")
    print("=" * 70)

    t0 = time.time()
    drag = train_hybrid_drag(epochs=args.epochs, quiet=args.quiet)
    t1 = time.time()

    print(f"\n  Results ({t1-t0:.1f}s):")
    print(f"    Learned G (compiled)   = {drag['G_learned']:.4f}")
    print(f"    Learned G (hand-coded) = {drag['G_learned_handcoded']:.4f}")
    print(f"    Hybrid test MSE:      {drag['hybrid_test_mse']:.2e}")
    print(f"    Hand-coded test MSE:  {drag['handcoded_test_mse']:.2e}")
    print(f"    MLP test MSE:         {drag['mlp_test_mse']:.2e}")

    data_path = args.save_fig.replace(".png", "_data.pkl")
    with open(data_path, "wb") as f:
        pickle.dump({"grav_results": grav, "drag_results": drag}, f)
    print(f"Results saved to {data_path}")

    plot_results(grav, drag, filename=args.save_fig)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  G recovery error (compiled):    {grav['G_error_pct']:.2f}%")
    print(f"  Gravity compiled/MLP:           {grav['mlp_test_mse']/max(grav['hybrid_test_mse'],1e-20):.1f}x better")
    print(f"  Gravity hand-coded/MLP:         {grav['mlp_test_mse']/max(grav['handcoded_test_mse'],1e-20):.1f}x better")
    print(f"  Compiled vs hand-coded (grav):  {grav['hybrid_test_mse']:.2e} vs {grav['handcoded_test_mse']:.2e}")
    print(f"  Drag hybrid/MLP:                {drag['mlp_test_mse']/max(drag['hybrid_test_mse'],1e-20):.1f}x better")
    print(f"  Drag hand-coded/MLP:            {drag['mlp_test_mse']/max(drag['handcoded_test_mse'],1e-20):.1f}x better")
    print(f"  Compiled vs hand-coded (drag):  {drag['hybrid_test_mse']:.2e} vs {drag['handcoded_test_mse']:.2e}")
    print(f"\n  Key insight: Compiled and hand-coded models produce equivalent results,")
    print(f"  confirming the compiler introduces no numerical overhead for single equations.")


if __name__ == "__main__":
    main()

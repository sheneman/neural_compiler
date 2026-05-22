############################################################
#
# Neural Compiler: Compiling scheme into composable and
#                  differentiable neural network representations
#
# pde_heat_equation.py: Compiled 1D heat equation PDE stepper learning thermal diffusivity
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""1D Heat Equation PDE: compiled diffusion structure + learned parameters.

Demonstrates the neural compiler on a PDE (partial differential equation):

    u_t = alpha * nabla^2 u  (1D heat equation)

Discretized as:  u_new = u + dt * alpha * L @ u
where L is the tridiagonal Laplacian matrix (1, -2, 1)/dx^2.

Five model types:
  1. Compiled — compile PDE stepper from Scheme, learn alpha
  2. Hand-coded PyTorch — same u + dt*alpha*L@u as nn.Module (no compiler)
  3. Hybrid compiled — compiled diffusion + MLP source
  4. Hybrid hand-coded — hand-coded diffusion + MLP source
  5. PINN — MLP approximates u(x,t) with PDE residual as soft loss
  6. Pure MLP — learn the time stepper u -> u_next from scratch

Usage:
    python examples/pde_heat_equation.py [--epochs 2000] [--quiet] [--save-fig PATH]
    python examples/pde_heat_equation.py --plot-only [--save-fig PATH]
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
# Grid parameters
# ---------------------------------------------------------------------------

N_GRID = 10
DX = 1.0 / (N_GRID - 1)
TRUE_ALPHA = 0.01
DT = 0.1
N_STEPS_TRAIN = 5
N_STEPS_EXTRAP = 20


# ---------------------------------------------------------------------------
# Analytical / reference PDE solver (numpy)
# ---------------------------------------------------------------------------

def make_laplacian(n, dx):
    L = np.zeros((n, n))
    for i in range(1, n - 1):
        L[i, i - 1] = 1.0 / dx ** 2
        L[i, i] = -2.0 / dx ** 2
        L[i, i + 1] = 1.0 / dx ** 2
    return L


def solve_heat_numpy(u0, alpha, dt, n_steps, dx):
    L = make_laplacian(len(u0), dx)
    u = u0.copy()
    trajectory = [u.copy()]
    for _ in range(n_steps):
        u = u + dt * alpha * L @ u
        trajectory.append(u.copy())
    return np.array(trajectory)


# ---------------------------------------------------------------------------
# Compiled PDE stepper
# ---------------------------------------------------------------------------

def build_heat_step_model():
    """Build a compiled single-step heat equation update.

    u_new = u + dt * alpha * (matvec L u)

    Inputs: u (vector), L (matrix), alpha (scalar), dt (scalar)
    """
    source = "(+ u (scale (* dt alpha) (matvec L u)))"
    graph = compile_scheme(source, inputs={
        "u": None, "L": None, "alpha": None, "dt": None,
    })
    return DirectModule(graph)


def make_laplacian_tensor(n, dx):
    L = torch.zeros(n, n)
    for i in range(1, n - 1):
        L[i, i - 1] = 1.0 / dx ** 2
        L[i, i] = -2.0 / dx ** 2
        L[i, i + 1] = 1.0 / dx ** 2
    return L


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class HeatCompiledAlpha(nn.Module):
    """Compiled PDE structure with trainable diffusivity alpha."""

    def __init__(self, n_grid, dx, dt):
        super().__init__()
        self.step_model = build_heat_step_model()
        self.register_buffer("L", make_laplacian_tensor(n_grid, dx))
        self.dt = dt
        self.alpha = nn.Parameter(torch.tensor(0.005))

    def step(self, u):
        return self.step_model({
            "u": u, "L": self.L, "alpha": self.alpha, "dt": torch.tensor(self.dt),
        })

    def rollout(self, u0, n_steps):
        u = u0
        trajectory = [u]
        for _ in range(n_steps):
            u = self.step(u)
            trajectory.append(u)
        return torch.stack(trajectory)


class HeatHybrid(nn.Module):
    """Compiled diffusion + learned source term."""

    def __init__(self, n_grid, dx, dt):
        super().__init__()
        self.step_model = build_heat_step_model()
        self.register_buffer("L", make_laplacian_tensor(n_grid, dx))
        self.dt = dt
        self.alpha = nn.Parameter(torch.tensor(0.005))
        self.source_net = nn.Sequential(
            nn.Linear(n_grid, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, n_grid),
        )

    def step(self, u):
        u_diff = self.step_model({
            "u": u, "L": self.L, "alpha": self.alpha, "dt": torch.tensor(self.dt),
        })
        return u_diff + self.dt * self.source_net(u)

    def rollout(self, u0, n_steps):
        u = u0
        trajectory = [u]
        for _ in range(n_steps):
            u = self.step(u)
            trajectory.append(u)
        return torch.stack(trajectory)


class HandCodedHeat(nn.Module):
    """Hand-coded PDE stepper: same physics as compiled, plain PyTorch."""

    def __init__(self, n_grid, dx, dt):
        super().__init__()
        self.register_buffer("L", make_laplacian_tensor(n_grid, dx))
        self.dt = dt
        self.alpha = nn.Parameter(torch.tensor(0.005))

    def step(self, u):
        return u + self.dt * self.alpha * torch.matmul(self.L, u)

    def rollout(self, u0, n_steps):
        u = u0
        trajectory = [u]
        for _ in range(n_steps):
            u = self.step(u)
            trajectory.append(u)
        return torch.stack(trajectory)


class HandCodedHybrid(nn.Module):
    """Hand-coded diffusion + learned source term (no compiler)."""

    def __init__(self, n_grid, dx, dt):
        super().__init__()
        self.register_buffer("L", make_laplacian_tensor(n_grid, dx))
        self.dt = dt
        self.alpha = nn.Parameter(torch.tensor(0.005))
        self.source_net = nn.Sequential(
            nn.Linear(n_grid, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, n_grid),
        )

    def step(self, u):
        u_diff = u + self.dt * self.alpha * torch.matmul(self.L, u)
        return u_diff + self.dt * self.source_net(u)

    def rollout(self, u0, n_steps):
        u = u0
        trajectory = [u]
        for _ in range(n_steps):
            u = self.step(u)
            trajectory.append(u)
        return torch.stack(trajectory)


class HeatPINN(nn.Module):
    """PINN baseline: MLP approximates u(x,t), PDE residual as soft loss."""

    def __init__(self, n_grid, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.alpha = nn.Parameter(torch.tensor(0.005))
        self.n_grid = n_grid

    def forward(self, x, t):
        xt = torch.stack([x, t], dim=-1)
        return self.net(xt).squeeze(-1)

    def pde_residual(self, x, t):
        x = x.requires_grad_(True)
        t = t.requires_grad_(True)
        u = self.forward(x, t)
        du_dt = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
        du_dx = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        d2u_dx2 = torch.autograd.grad(du_dx.sum(), x, create_graph=True)[0]
        return du_dt - self.alpha * d2u_dx2

    def predict_trajectory(self, u0, x_grid, dt, n_steps):
        trajectory = [u0]
        for step in range(1, n_steps + 1):
            t_val = step * dt
            t_tensor = torch.full_like(x_grid, t_val)
            u = self.forward(x_grid, t_tensor)
            trajectory.append(u)
        return torch.stack(trajectory)


class HeatMLP(nn.Module):
    """Pure MLP baseline: learn the time step map u -> u_next."""

    def __init__(self, n_grid, hidden=64, layers=3):
        super().__init__()
        mods = [nn.Linear(n_grid, hidden), nn.ReLU()]
        for _ in range(layers - 1):
            mods.extend([nn.Linear(hidden, hidden), nn.ReLU()])
        mods.append(nn.Linear(hidden, n_grid))
        self.net = nn.Sequential(*mods)

    def step(self, u):
        return self.net(u)

    def rollout(self, u0, n_steps):
        u = u0
        trajectory = [u]
        for _ in range(n_steps):
            u = self.step(u)
            trajectory.append(u)
        return torch.stack(trajectory)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_heat_data(n_samples, n_grid, dx, alpha, dt, n_steps, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    u0_list = []
    traj_list = []
    for _ in range(n_samples):
        u0 = np.zeros(n_grid)
        n_modes = np.random.randint(1, 4)
        for _ in range(n_modes):
            k = np.random.randint(1, 5)
            amp = np.random.uniform(0.5, 2.0)
            x = np.linspace(0, 1, n_grid)
            u0 += amp * np.sin(k * np.pi * x)
        traj = solve_heat_numpy(u0, alpha, dt, n_steps, dx)
        u0_list.append(torch.tensor(u0, dtype=torch.float32))
        traj_list.append(torch.tensor(traj, dtype=torch.float32))

    return u0_list, traj_list


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_pinn(pinn, u0_list, traj_list, x_grid, dt, n_steps, epochs,
               u0_test=None, traj_test=None, quiet=False):
    """Train PINN with data loss + PDE residual loss."""
    opt = torch.optim.Adam(pinn.parameters(), lr=0.001)
    history = []
    test_loss_history = {"pinn_test_loss": [], "pinn_test_epochs": []}
    n_train = len(u0_list)
    n_colloc = 200
    eval_interval = max(1, epochs // 100)

    for epoch in range(epochs):
        opt.zero_grad()

        # Data loss: match observed trajectories
        data_loss = torch.tensor(0.0)
        for u0, traj in zip(u0_list, traj_list):
            for step_idx in range(n_steps + 1):
                t_val = step_idx * dt
                t_tensor = torch.full_like(x_grid, t_val)
                u_pred = pinn.forward(x_grid, t_tensor)
                data_loss = data_loss + ((u_pred - traj[step_idx]) ** 2).mean()
        data_loss = data_loss / (n_train * (n_steps + 1))

        # Physics loss: PDE residual at random collocation points
        x_col = torch.rand(n_colloc, requires_grad=True)
        t_col = torch.rand(n_colloc, requires_grad=True) * (n_steps * dt)
        residual = pinn.pde_residual(x_col, t_col)
        phys_loss = (residual ** 2).mean()

        loss = data_loss + 0.1 * phys_loss
        loss.backward()
        opt.step()
        history.append(loss.item())

        # Periodic test loss evaluation
        if u0_test is not None and traj_test is not None and (epoch + 1) % eval_interval == 0:
            with torch.no_grad():
                test_loss_p = 0.0
                for u0_t, traj_t in zip(u0_test, traj_test):
                    pred_p = pinn.predict_trajectory(u0_t, x_grid, dt, n_steps)
                    test_loss_p += ((pred_p - traj_t[:n_steps+1]) ** 2).mean().item()
                test_loss_p /= len(u0_test)
            test_loss_history["pinn_test_loss"].append(test_loss_p)
            test_loss_history["pinn_test_epochs"].append(epoch + 1)

        if not quiet and (epoch + 1) % 500 == 0:
            print(f"  PINN Epoch {epoch+1:4d}  total={loss.item():.2e}  "
                  f"data={data_loss.item():.2e}  phys={phys_loss.item():.2e}  "
                  f"alpha={pinn.alpha.item():.6f}")

    return history, test_loss_history


def train_alpha_recovery(epochs=2000, n_train=50, quiet=False):
    u0_list, traj_list = generate_heat_data(
        n_train, N_GRID, DX, TRUE_ALPHA, DT, N_STEPS_TRAIN,
    )
    u0_test, traj_test = generate_heat_data(
        20, N_GRID, DX, TRUE_ALPHA, DT, N_STEPS_EXTRAP, seed=99,
    )

    model = HeatCompiledAlpha(N_GRID, DX, DT)
    handcoded = HandCodedHeat(N_GRID, DX, DT)
    pinn = HeatPINN(N_GRID)
    mlp = HeatMLP(N_GRID)

    opt_c = torch.optim.Adam(model.parameters(), lr=0.001)
    opt_h = torch.optim.Adam(handcoded.parameters(), lr=0.001)
    opt_m = torch.optim.Adam(mlp.parameters(), lr=0.001)

    history = {
        "compiled_loss": [], "handcoded_loss": [], "mlp_loss": [],
        "compiled_test_loss": [], "handcoded_test_loss": [], "mlp_test_loss": [],
        "test_epochs": [],
        "alpha_est": [], "alpha_handcoded": [],
    }

    eval_interval = max(1, epochs // 100)

    for epoch in range(epochs):
        # Compiled model
        opt_c.zero_grad()
        loss_c = torch.tensor(0.0)
        for u0, traj in zip(u0_list, traj_list):
            pred = model.rollout(u0, N_STEPS_TRAIN)
            loss_c = loss_c + ((pred - traj) ** 2).mean()
        loss_c = loss_c / n_train
        loss_c.backward()
        opt_c.step()

        # Hand-coded model
        opt_h.zero_grad()
        loss_h = torch.tensor(0.0)
        for u0, traj in zip(u0_list, traj_list):
            pred = handcoded.rollout(u0, N_STEPS_TRAIN)
            loss_h = loss_h + ((pred - traj) ** 2).mean()
        loss_h = loss_h / n_train
        loss_h.backward()
        opt_h.step()

        # MLP
        opt_m.zero_grad()
        loss_m = torch.tensor(0.0)
        for u0, traj in zip(u0_list, traj_list):
            pred = mlp.rollout(u0, N_STEPS_TRAIN)
            loss_m = loss_m + ((pred - traj) ** 2).mean()
        loss_m = loss_m / n_train
        loss_m.backward()
        opt_m.step()

        history["compiled_loss"].append(loss_c.item())
        history["handcoded_loss"].append(loss_h.item())
        history["mlp_loss"].append(loss_m.item())
        history["alpha_est"].append(model.alpha.item())
        history["alpha_handcoded"].append(handcoded.alpha.item())

        # Periodic test loss evaluation
        if (epoch + 1) % eval_interval == 0:
            with torch.no_grad():
                test_c = test_h = test_m = 0.0
                n_test_samples = len(u0_test)
                for u0_t, traj_t in zip(u0_test, traj_test):
                    pred_c = model.rollout(u0_t, N_STEPS_TRAIN)
                    pred_h = handcoded.rollout(u0_t, N_STEPS_TRAIN)
                    pred_m = mlp.rollout(u0_t, N_STEPS_TRAIN)
                    test_c += ((pred_c - traj_t[:N_STEPS_TRAIN+1]) ** 2).mean().item()
                    test_h += ((pred_h - traj_t[:N_STEPS_TRAIN+1]) ** 2).mean().item()
                    test_m += ((pred_m - traj_t[:N_STEPS_TRAIN+1]) ** 2).mean().item()
                test_c /= n_test_samples
                test_h /= n_test_samples
                test_m /= n_test_samples
            history["compiled_test_loss"].append(test_c)
            history["handcoded_test_loss"].append(test_h)
            history["mlp_test_loss"].append(test_m)
            history["test_epochs"].append(epoch + 1)

        if not quiet and (epoch + 1) % 500 == 0:
            print(f"  Epoch {epoch+1:4d}  compiled={loss_c.item():.2e}  "
                  f"handcoded={loss_h.item():.2e}  mlp={loss_m.item():.2e}  "
                  f"alpha_c={model.alpha.item():.6f}  alpha_h={handcoded.alpha.item():.6f}")

    # Train PINN separately (different training paradigm)
    if not quiet:
        print("\n  Training PINN baseline...")
    x_grid = torch.linspace(0, 1, N_GRID)
    pinn_history, pinn_test_history = train_pinn(
        pinn, u0_list, traj_list, x_grid, DT, N_STEPS_TRAIN, epochs,
        u0_test=u0_test, traj_test=traj_test, quiet=quiet)

    # Test: interpolation (same n_steps) and extrapolation (longer horizon)
    with torch.no_grad():
        interp_c = interp_h = interp_m = interp_p = 0.0
        extrap_c = extrap_h = extrap_m = extrap_p = 0.0
        x_grid_test = torch.linspace(0, 1, N_GRID)

        for u0, traj in zip(u0_test, traj_test):
            pred_c = model.rollout(u0, N_STEPS_TRAIN)
            pred_h = handcoded.rollout(u0, N_STEPS_TRAIN)
            pred_m = mlp.rollout(u0, N_STEPS_TRAIN)
            pred_p = pinn.predict_trajectory(u0, x_grid_test, DT, N_STEPS_TRAIN)
            interp_c += ((pred_c - traj[:N_STEPS_TRAIN+1]) ** 2).mean().item()
            interp_h += ((pred_h - traj[:N_STEPS_TRAIN+1]) ** 2).mean().item()
            interp_m += ((pred_m - traj[:N_STEPS_TRAIN+1]) ** 2).mean().item()
            interp_p += ((pred_p - traj[:N_STEPS_TRAIN+1]) ** 2).mean().item()

            pred_c_ext = model.rollout(u0, N_STEPS_EXTRAP)
            pred_h_ext = handcoded.rollout(u0, N_STEPS_EXTRAP)
            pred_m_ext = mlp.rollout(u0, N_STEPS_EXTRAP)
            pred_p_ext = pinn.predict_trajectory(u0, x_grid_test, DT, N_STEPS_EXTRAP)
            extrap_c += ((pred_c_ext - traj) ** 2).mean().item()
            extrap_h += ((pred_h_ext - traj) ** 2).mean().item()
            extrap_m += ((pred_m_ext - traj) ** 2).mean().item()
            extrap_p += ((pred_p_ext - traj) ** 2).mean().item()

        n_test = len(u0_test)
        interp_c /= n_test
        interp_h /= n_test
        interp_m /= n_test
        interp_p /= n_test
        extrap_c /= n_test
        extrap_h /= n_test
        extrap_m /= n_test
        extrap_p /= n_test

    return {
        "alpha_learned": model.alpha.item(),
        "alpha_handcoded": handcoded.alpha.item(),
        "alpha_pinn": pinn.alpha.item(),
        "alpha_true": TRUE_ALPHA,
        "alpha_error_pct": abs(model.alpha.item() - TRUE_ALPHA) / TRUE_ALPHA * 100,
        "alpha_hc_error_pct": abs(handcoded.alpha.item() - TRUE_ALPHA) / TRUE_ALPHA * 100,
        "alpha_pinn_error_pct": abs(pinn.alpha.item() - TRUE_ALPHA) / TRUE_ALPHA * 100,
        "interp_compiled": interp_c,
        "interp_handcoded": interp_h,
        "interp_mlp": interp_m,
        "interp_pinn": interp_p,
        "extrap_compiled": extrap_c,
        "extrap_handcoded": extrap_h,
        "extrap_mlp": extrap_m,
        "extrap_pinn": extrap_p,
        "history": history,
        "pinn_history": pinn_history,
        "pinn_test_history": pinn_test_history,
        "params_compiled": 1,
        "params_handcoded": 1,
        "params_pinn": sum(p.numel() for p in pinn.parameters()),
        "params_mlp": sum(p.numel() for p in mlp.parameters()),
    }


def train_hybrid_source(epochs=2000, n_train=50, quiet=False):
    """Train on data that has diffusion + a heat source term."""
    source_fn = lambda x: 0.5 * np.sin(2 * np.pi * x)

    np.random.seed(42)
    torch.manual_seed(42)

    u0_list, traj_list = [], []
    for _ in range(n_train):
        u0 = np.zeros(N_GRID)
        n_modes = np.random.randint(1, 3)
        for _ in range(n_modes):
            k = np.random.randint(1, 4)
            amp = np.random.uniform(0.5, 2.0)
            x = np.linspace(0, 1, N_GRID)
            u0 += amp * np.sin(k * np.pi * x)

        L_np = make_laplacian(N_GRID, DX)
        x_grid = np.linspace(0, 1, N_GRID)
        source = source_fn(x_grid)
        u = u0.copy()
        traj = [u.copy()]
        for _ in range(N_STEPS_TRAIN):
            u = u + DT * (TRUE_ALPHA * L_np @ u + source)
            traj.append(u.copy())

        u0_list.append(torch.tensor(u0, dtype=torch.float32))
        traj_list.append(torch.tensor(np.array(traj), dtype=torch.float32))

    hybrid = HeatHybrid(N_GRID, DX, DT)
    hc_hybrid = HandCodedHybrid(N_GRID, DX, DT)
    mlp = HeatMLP(N_GRID)
    opt_h = torch.optim.Adam(hybrid.parameters(), lr=0.001)
    opt_hc = torch.optim.Adam(hc_hybrid.parameters(), lr=0.001)
    opt_m = torch.optim.Adam(mlp.parameters(), lr=0.001)

    history = {"hybrid_loss": [], "hc_hybrid_loss": [], "mlp_loss": []}

    for epoch in range(epochs):
        opt_h.zero_grad()
        loss_h = torch.tensor(0.0)
        for u0, traj in zip(u0_list, traj_list):
            pred = hybrid.rollout(u0, N_STEPS_TRAIN)
            loss_h = loss_h + ((pred - traj) ** 2).mean()
        loss_h = loss_h / n_train
        loss_h.backward()
        opt_h.step()

        opt_hc.zero_grad()
        loss_hc = torch.tensor(0.0)
        for u0, traj in zip(u0_list, traj_list):
            pred = hc_hybrid.rollout(u0, N_STEPS_TRAIN)
            loss_hc = loss_hc + ((pred - traj) ** 2).mean()
        loss_hc = loss_hc / n_train
        loss_hc.backward()
        opt_hc.step()

        opt_m.zero_grad()
        loss_m = torch.tensor(0.0)
        for u0, traj in zip(u0_list, traj_list):
            pred = mlp.rollout(u0, N_STEPS_TRAIN)
            loss_m = loss_m + ((pred - traj) ** 2).mean()
        loss_m = loss_m / n_train
        loss_m.backward()
        opt_m.step()

        history["hybrid_loss"].append(loss_h.item())
        history["hc_hybrid_loss"].append(loss_hc.item())
        history["mlp_loss"].append(loss_m.item())

        if not quiet and (epoch + 1) % 500 == 0:
            print(f"  Epoch {epoch+1:4d}  hybrid={loss_h.item():.2e}  "
                  f"hc_hybrid={loss_hc.item():.2e}  mlp={loss_m.item():.2e}  "
                  f"alpha_c={hybrid.alpha.item():.6f}  alpha_h={hc_hybrid.alpha.item():.6f}")

    return {
        "alpha_learned": hybrid.alpha.item(),
        "alpha_hc_hybrid": hc_hybrid.alpha.item(),
        "hybrid_final_loss": history["hybrid_loss"][-1] if history["hybrid_loss"] else float("inf"),
        "hc_hybrid_final_loss": history["hc_hybrid_loss"][-1] if history["hc_hybrid_loss"] else float("inf"),
        "mlp_final_loss": history["mlp_loss"][-1] if history["mlp_loss"] else float("inf"),
        "history": history,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(alpha_results, hybrid_results, filename="examples/pde_heat_equation.png"):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Alpha recovery
    ax = axes[0, 0]
    epochs = range(1, len(alpha_results["history"]["alpha_est"]) + 1)
    ax.plot(epochs, alpha_results["history"]["alpha_est"], "b-", linewidth=2, label="Compiled")
    ax.plot(epochs, alpha_results["history"]["alpha_handcoded"], color="#9C27B0",
            linewidth=2, linestyle="--", label="Hand-coded", alpha=0.8)
    ax.axhline(y=TRUE_ALPHA, color="r", linestyle=":", linewidth=1.5,
               label=f"True alpha = {TRUE_ALPHA}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learned alpha")
    ax.set_title(f"Thermal Diffusivity Recovery\n"
                 f"Compiled: {alpha_results['alpha_learned']:.6f} "
                 f"({alpha_results['alpha_error_pct']:.2f}% err), "
                 f"PINN: {alpha_results['alpha_pinn']:.6f} "
                 f"({alpha_results['alpha_pinn_error_pct']:.2f}% err)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Test loss comparison
    ax = axes[0, 1]
    test_epochs = alpha_results["history"]["test_epochs"]
    ax.semilogy(test_epochs, alpha_results["history"]["compiled_test_loss"],
                label="Compiled (1 param)", linewidth=2, color="steelblue")
    ax.semilogy(test_epochs, alpha_results["history"]["handcoded_test_loss"],
                label="Hand-coded (1 param)", linewidth=2, color="#9C27B0",
                linestyle="--", alpha=0.8)
    pinn_test = alpha_results["pinn_test_history"]
    ax.semilogy(pinn_test["pinn_test_epochs"], pinn_test["pinn_test_loss"],
                label=f"PINN ({alpha_results['params_pinn']} params)",
                linewidth=2, color="#795548", alpha=0.7)
    ax.semilogy(test_epochs, alpha_results["history"]["mlp_test_loss"],
                label=f"MLP ({alpha_results['params_mlp']} params)",
                linewidth=2, color="coral", alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test MSE")
    ax.set_title("Diffusion-only: Test Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Interpolation vs Extrapolation
    ax = axes[1, 0]
    labels = ["Interp\n(5 steps)", "Extrap\n(20 steps)"]
    compiled_vals = [alpha_results["interp_compiled"], alpha_results["extrap_compiled"]]
    hc_vals = [alpha_results["interp_handcoded"], alpha_results["extrap_handcoded"]]
    pinn_vals = [alpha_results["interp_pinn"], alpha_results["extrap_pinn"]]
    mlp_vals = [alpha_results["interp_mlp"], alpha_results["extrap_mlp"]]
    x = np.arange(len(labels))
    w = 0.2
    ax.bar(x - 1.5*w, compiled_vals, w, label="Compiled", color="steelblue")
    ax.bar(x - 0.5*w, hc_vals, w, label="Hand-coded", color="#9C27B0")
    ax.bar(x + 0.5*w, pinn_vals, w, label="PINN", color="#795548")
    ax.bar(x + 1.5*w, mlp_vals, w, label="MLP", color="coral")
    ax.set_ylabel("Test MSE")
    ax.set_title("Interpolation vs Extrapolation")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=8)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")

    # 4. Hybrid source training
    ax = axes[1, 1]
    ax.semilogy(hybrid_results["history"]["hybrid_loss"],
                label="Compiled hybrid", linewidth=2, color="steelblue")
    ax.semilogy(hybrid_results["history"]["hc_hybrid_loss"],
                label="Hand-coded hybrid", linewidth=2, color="#9C27B0",
                linestyle="--", alpha=0.8)
    ax.semilogy(hybrid_results["history"]["mlp_loss"],
                label="Pure MLP", linewidth=2, color="coral", alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"Diffusion + Source: Hybrid vs MLP\n"
                 f"alpha compiled: {hybrid_results['alpha_learned']:.6f}, "
                 f"hand-coded: {hybrid_results['alpha_hc_hybrid']:.6f}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {filename}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="1D Heat Equation PDE Demo")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--save-fig", type=str, default="examples/pde_heat_equation.png",
                        help="Path for the output figure (default: examples/pde_heat_equation.png)")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip training; load saved data and regenerate figures")
    args = parser.parse_args()

    fig_path = args.save_fig
    data_path = fig_path.rsplit(".", 1)[0] + "_data.pkl" if "." in fig_path else fig_path + "_data.pkl"

    if args.plot_only:
        print(f"Loading saved data from {data_path} ...")
        with open(data_path, "rb") as f:
            saved = pickle.load(f)
        plot_results(saved["alpha_results"], saved["hybrid_results"], filename=fig_path)
        print("Done.")
        return

    print("=" * 70)
    print("1D HEAT EQUATION PDE — NEURAL COMPILER DEMO")
    print(f"  Grid: {N_GRID} points, dx={DX:.4f}, dt={DT}")
    print(f"  True thermal diffusivity: alpha = {TRUE_ALPHA}")
    print("=" * 70)

    # Experiment 1: alpha recovery
    print("\nEXPERIMENT 1: Thermal Diffusivity Recovery")
    print("  Compiled:   u_new = u + dt*alpha*(L@u), learn alpha (from Scheme)")
    print("  Hand-coded: same equation as nn.Module (plain PyTorch)")
    print("  PINN:       MLP u(x,t) with PDE residual as soft loss")
    print("  MLP:        learns u -> u_next from scratch")
    print("-" * 50)

    t0 = time.time()
    alpha_res = train_alpha_recovery(epochs=args.epochs, quiet=args.quiet)
    t1 = time.time()

    print(f"\n  Results ({t1-t0:.1f}s):")
    print(f"    Alpha recovery:")
    print(f"      Compiled:   {alpha_res['alpha_learned']:.6f}  "
          f"(error: {alpha_res['alpha_error_pct']:.2f}%)")
    print(f"      Hand-coded: {alpha_res['alpha_handcoded']:.6f}  "
          f"(error: {alpha_res['alpha_hc_error_pct']:.2f}%)")
    print(f"      PINN:       {alpha_res['alpha_pinn']:.6f}  "
          f"(error: {alpha_res['alpha_pinn_error_pct']:.2f}%)")
    print(f"    Parameters: compiled={alpha_res['params_compiled']}, "
          f"hand-coded={alpha_res['params_handcoded']}, "
          f"PINN={alpha_res['params_pinn']}, MLP={alpha_res['params_mlp']}")
    print(f"    Interpolation MSE:")
    print(f"      Compiled: {alpha_res['interp_compiled']:.2e}  "
          f"Hand-coded: {alpha_res['interp_handcoded']:.2e}  "
          f"PINN: {alpha_res['interp_pinn']:.2e}  "
          f"MLP: {alpha_res['interp_mlp']:.2e}")
    print(f"    Extrapolation MSE:")
    print(f"      Compiled: {alpha_res['extrap_compiled']:.2e}  "
          f"Hand-coded: {alpha_res['extrap_handcoded']:.2e}  "
          f"PINN: {alpha_res['extrap_pinn']:.2e}  "
          f"MLP: {alpha_res['extrap_mlp']:.2e}")

    # Experiment 2: hybrid with source
    print()
    print("EXPERIMENT 2: Hybrid Diffusion + Source Term")
    print("  Compiled hybrid:   compiled diffusion + MLP source")
    print("  Hand-coded hybrid: hand-coded diffusion + MLP source")
    print("  MLP:               learns full dynamics from scratch")
    print("-" * 50)

    t0 = time.time()
    hybrid_res = train_hybrid_source(epochs=args.epochs, quiet=args.quiet)
    t1 = time.time()

    print(f"\n  Results ({t1-t0:.1f}s):")
    print(f"    Alpha: compiled={hybrid_res['alpha_learned']:.6f}  "
          f"hand-coded={hybrid_res['alpha_hc_hybrid']:.6f}")
    print(f"    Final loss — compiled hybrid: {hybrid_res['hybrid_final_loss']:.2e}  "
          f"hand-coded hybrid: {hybrid_res['hc_hybrid_final_loss']:.2e}  "
          f"MLP: {hybrid_res['mlp_final_loss']:.2e}")

    # Save data for later re-plotting
    saved_data = {
        "alpha_results": alpha_res,
        "hybrid_results": hybrid_res,
    }
    with open(data_path, "wb") as f:
        pickle.dump(saved_data, f)
    print(f"\n  Plot data saved to {data_path}")

    plot_results(alpha_res, hybrid_res, filename=fig_path)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Alpha recovery (compiled):      {alpha_res['alpha_error_pct']:.2f}% error")
    print(f"  Alpha recovery (hand-coded):    {alpha_res['alpha_hc_error_pct']:.2f}% error")
    print(f"  Alpha recovery (PINN):          {alpha_res['alpha_pinn_error_pct']:.2f}% error")
    print(f"  Parameters: compiled={alpha_res['params_compiled']}, "
          f"hand-coded={alpha_res['params_handcoded']}, "
          f"PINN={alpha_res['params_pinn']}, MLP={alpha_res['params_mlp']}")

    print(f"\n  Compiled vs hand-coded: numerically identical (both hard constraints)")
    print(f"  Compiler advantage: systematic generation from symbolic PDE specification")
    print(f"  PINN: soft physics constraints — worse alpha recovery, less extrapolation")
    print(f"  MLP: no physics — poor extrapolation to longer time horizons")


if __name__ == "__main__":
    main()

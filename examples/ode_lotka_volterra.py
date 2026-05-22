############################################################
#
# Neural Compiler: Compiling scheme into composable and
#                  differentiable neural network representations
#
# ode_lotka_volterra.py: Compiled Lotka-Volterra predator-prey ODE with RK4 solver learning rate constants
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Partially-known ODE systems: Lotka-Volterra predator-prey dynamics.

Demonstrates two scenarios for integrating compiled subgraphs with ODE solvers:

Scenario 1 — Known structure, unknown parameters:
  Compile the full Lotka-Volterra RHS as frozen compiled modules.
  Learn the 4 rate constants (alpha, beta, delta, gamma) via gradient flow
  through the RK4 integrator and frozen subgraphs.

Scenario 2 — Known interaction + unknown growth/death:
  Compile only the predation term (beta * x * y) as a frozen subgraph.
  Learn the growth and death terms with small MLPs.
  The compiled subgraph provides the exact bilinear interaction;
  the MLPs discover the linear growth/death dynamics.

Baselines:
  - Pure MLP: learns the full RHS from scratch
  - Neural ODE (torchdiffeq): continuous-depth MLP, adaptive step solver

All models train on noisy trajectory snapshots via multiple shooting and
are evaluated on in-distribution reconstruction, long-horizon extrapolation,
parameter recovery, and noise robustness.
"""

import argparse
import math
import time
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_compiler.compiler import compile_scheme
from neural_compiler.evaluator.direct_module import DirectModule


# ---------------------------------------------------------------------------
# True system parameters
# ---------------------------------------------------------------------------

TRUE_PARAMS = {
    "alpha": 1.0,    # prey growth rate
    "beta": 0.5,     # predation rate
    "delta": 0.25,   # predator growth from predation
    "gamma": 0.5,    # predator death rate
}

X0_DEFAULT = torch.tensor([4.0, 1.0])  # (prey, predator) initial condition


# ---------------------------------------------------------------------------
# Ground truth ODE RHS (for data generation)
# ---------------------------------------------------------------------------

def lotka_volterra_rhs(state: torch.Tensor, params: dict[str, float]) -> torch.Tensor:
    x, y = state[..., 0], state[..., 1]
    dx = params["alpha"] * x - params["beta"] * x * y
    dy = params["delta"] * x * y - params["gamma"] * y
    return torch.stack([dx, dy], dim=-1)


def rk4_step(rhs_fn, state, dt):
    k1 = rhs_fn(state)
    k2 = rhs_fn(state + 0.5 * dt * k1)
    k3 = rhs_fn(state + 0.5 * dt * k2)
    k4 = rhs_fn(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def generate_trajectory(params: dict[str, float], x0: torch.Tensor,
                        t_end: float, dt: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    def rhs(state):
        return lotka_volterra_rhs(state, params)

    steps = int(t_end / dt)
    trajectory = [x0]
    state = x0
    for _ in range(steps):
        state = rk4_step(rhs, state, dt)
        trajectory.append(state)
    traj = torch.stack(trajectory, dim=0)
    t = torch.linspace(0.0, t_end, traj.shape[0])
    return t, traj


def add_noise(traj: torch.Tensor, noise_level: float, seed: int = 42) -> torch.Tensor:
    torch.manual_seed(seed)
    scale = noise_level * traj.abs().mean()
    return traj + scale * torch.randn_like(traj)


# ---------------------------------------------------------------------------
# Compiled subgraphs
# ---------------------------------------------------------------------------

def compile_lv_rhs():
    prey_scheme = "(- (* alpha x) (* beta (* x y)))"
    pred_scheme = "(- (* delta (* x y)) (* gamma y))"

    prey_inputs = {"alpha": None, "beta": None, "x": None, "y": None}
    pred_inputs = {"delta": None, "gamma": None, "x": None, "y": None}

    g_prey = compile_scheme(prey_scheme, inputs=prey_inputs)
    m_prey = DirectModule(g_prey)
    for p in m_prey.parameters():
        p.requires_grad = False
    m_prey.eval()

    g_pred = compile_scheme(pred_scheme, inputs=pred_inputs)
    m_pred = DirectModule(g_pred)
    for p in m_pred.parameters():
        p.requires_grad = False
    m_pred.eval()

    print(f"  Compiled prey RHS: {len(g_prey.nodes)} nodes, depth={g_prey.depth()}")
    print(f"  Compiled predator RHS: {len(g_pred.nodes)} nodes, depth={g_pred.depth()}")

    return m_prey, m_pred


def compile_predation():
    scheme = "(* beta (* x y))"
    inputs = {"beta": None, "x": None, "y": None}
    g = compile_scheme(scheme, inputs=inputs)
    m = DirectModule(g)
    for p in m.parameters():
        p.requires_grad = False
    m.eval()
    print(f"  Compiled predation term: {len(g.nodes)} nodes, depth={g.depth()}")
    return m


# ---------------------------------------------------------------------------
# Model: Scenario 1 — Known structure, unknown parameters
# ---------------------------------------------------------------------------

class LVKnownStructure(nn.Module):
    """Full LV structure compiled; learn 4 rate constants."""

    def __init__(self, m_prey: DirectModule, m_pred: DirectModule):
        super().__init__()
        self.m_prey = m_prey
        self.m_pred = m_pred
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.delta = nn.Parameter(torch.tensor(0.5))
        self.gamma = nn.Parameter(torch.tensor(0.5))

    def rhs(self, state):
        """state: (batch, 2) or (2,)"""
        if state.dim() == 1:
            state = state.unsqueeze(0)
        x, y = state[:, 0], state[:, 1]
        batch = x.shape[0]

        prey_inputs = {
            "alpha": self.alpha.expand(batch),
            "beta": self.beta.expand(batch),
            "x": x, "y": y,
        }
        pred_inputs = {
            "delta": self.delta.expand(batch),
            "gamma": self.gamma.expand(batch),
            "x": x, "y": y,
        }

        dx = self.m_prey.forward_batch(prey_inputs)
        dy = self.m_pred.forward_batch(pred_inputs)
        return torch.stack([dx, dy], dim=-1)

    def integrate(self, x0, t_end, dt):
        steps = int(t_end / dt)
        trajectory = [x0.unsqueeze(0)]
        state = x0.unsqueeze(0)
        for _ in range(steps):
            state = rk4_step(self.rhs, state, dt)
            trajectory.append(state)
        return torch.cat(trajectory, dim=0).squeeze(1)

    def learned_params(self):
        return {
            "alpha": self.alpha.item(),
            "beta": self.beta.item(),
            "delta": self.delta.item(),
            "gamma": self.gamma.item(),
        }


# ---------------------------------------------------------------------------
# Model: Scenario 2 — Known interaction + learned growth/death
# ---------------------------------------------------------------------------

class LVHybrid(nn.Module):
    """Compiled predation term + MLP growth/death."""

    def __init__(self, m_predation: DirectModule, hidden: int = 32):
        super().__init__()
        self.m_predation = m_predation
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.growth_net = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.death_net = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.growth_net[-1].weight)
        nn.init.constant_(self.growth_net[-1].bias, 1.0)
        nn.init.zeros_(self.death_net[-1].weight)
        nn.init.constant_(self.death_net[-1].bias, 0.5)

    def rhs(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        x, y = state[:, 0], state[:, 1]
        batch = x.shape[0]

        predation_inputs = {
            "beta": self.beta.expand(batch),
            "x": x, "y": y,
        }
        predation = self.m_predation.forward_batch(predation_inputs)

        xy = torch.stack([x, y], dim=-1)
        growth = self.growth_net(xy).squeeze(-1)
        death = self.death_net(xy).squeeze(-1)

        dx = growth - predation
        dy = predation - death
        return torch.stack([dx, dy], dim=-1)

    def integrate(self, x0, t_end, dt):
        steps = int(t_end / dt)
        trajectory = [x0.unsqueeze(0)]
        state = x0.unsqueeze(0)
        for _ in range(steps):
            state = rk4_step(self.rhs, state, dt)
            trajectory.append(state)
        return torch.cat(trajectory, dim=0).squeeze(1)


# ---------------------------------------------------------------------------
# Baseline: Pure MLP
# ---------------------------------------------------------------------------

class LVMLP(nn.Module):
    """Pure MLP learning the full RHS from scratch."""

    def __init__(self, hidden: int = 64, layers: int = 3):
        super().__init__()
        modules = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(layers - 1):
            modules.extend([nn.Linear(hidden, hidden), nn.Tanh()])
        modules.append(nn.Linear(hidden, 2))
        self.net = nn.Sequential(*modules)

    def rhs(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return self.net(state)

    def integrate(self, x0, t_end, dt):
        steps = int(t_end / dt)
        trajectory = [x0.unsqueeze(0)]
        state = x0.unsqueeze(0)
        for _ in range(steps):
            state = rk4_step(self.rhs, state, dt)
            trajectory.append(state)
        return torch.cat(trajectory, dim=0).squeeze(1)


# ---------------------------------------------------------------------------
# Baseline: Neural ODE
# ---------------------------------------------------------------------------

class LVHandCoded(nn.Module):
    """Hand-coded PyTorch: same LV equations, no compiler."""

    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.delta = nn.Parameter(torch.tensor(0.5))
        self.gamma = nn.Parameter(torch.tensor(0.5))

    def rhs(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        x, y = state[:, 0], state[:, 1]
        dx = self.alpha * x - self.beta * x * y
        dy = self.delta * x * y - self.gamma * y
        return torch.stack([dx, dy], dim=-1)

    def integrate(self, x0, t_end, dt):
        steps = int(t_end / dt)
        trajectory = [x0.unsqueeze(0)]
        state = x0.unsqueeze(0)
        for _ in range(steps):
            state = rk4_step(self.rhs, state, dt)
            trajectory.append(state)
        return torch.cat(trajectory, dim=0).squeeze(1)

    def learned_params(self):
        return {k: getattr(self, k).item() for k in ["alpha", "beta", "delta", "gamma"]}


# ---------------------------------------------------------------------------
# Baseline: PINN
# ---------------------------------------------------------------------------

class LVPINN(nn.Module):
    """PINN: MLP approximates trajectory, ODE residual as soft loss."""

    def __init__(self, hidden: int = 64, layers: int = 3):
        super().__init__()
        mods = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(layers - 1):
            mods.extend([nn.Linear(hidden, hidden), nn.Tanh()])
        mods.append(nn.Linear(hidden, 2))
        self.net = nn.Sequential(*mods)
        # Trainable physics parameters
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.delta = nn.Parameter(torch.tensor(0.5))
        self.gamma = nn.Parameter(torch.tensor(0.5))

    def forward(self, t):
        """t: (N,) -> (N, 2) predicted [x, y]."""
        return self.net(t.unsqueeze(-1) if t.dim() == 1 else t)

    def physics_loss(self, t_collocation):
        """ODE residual loss at collocation points."""
        t = t_collocation.unsqueeze(-1).requires_grad_(True)
        u = self.net(t)  # (N, 2)
        x, y = u[:, 0:1], u[:, 1:2]

        dx_dt = torch.autograd.grad(
            x, t, grad_outputs=torch.ones_like(x), create_graph=True
        )[0]
        dy_dt = torch.autograd.grad(
            y, t, grad_outputs=torch.ones_like(y), create_graph=True
        )[0]

        rhs_x = self.alpha * x - self.beta * x * y
        rhs_y = self.delta * x * y - self.gamma * y

        residual = (dx_dt - rhs_x).pow(2).mean() + (dy_dt - rhs_y).pow(2).mean()
        return residual

    def integrate(self, x0, t_end, dt):
        """Evaluate the trained PINN at discrete time points."""
        n_steps = int(t_end / dt) + 1
        t = torch.linspace(0.0, t_end, n_steps)
        with torch.no_grad():
            return self.forward(t)

    def learned_params(self):
        return {k: getattr(self, k).item() for k in ["alpha", "beta", "delta", "gamma"]}


# ---------------------------------------------------------------------------
# Baseline: Neural ODE
# ---------------------------------------------------------------------------

class NeuralODEFunc(nn.Module):
    def __init__(self, hidden: int = 64, layers: int = 3):
        super().__init__()
        modules = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(layers - 1):
            modules.extend([nn.Linear(hidden, hidden), nn.Tanh()])
        modules.append(nn.Linear(hidden, 2))
        self.net = nn.Sequential(*modules)

    def forward(self, t, y):
        return self.net(y)


class NeuralODE(nn.Module):
    def __init__(self, hidden: int = 64, layers: int = 3):
        super().__init__()
        self.func = NeuralODEFunc(hidden, layers)

    def rhs(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return self.func(torch.tensor(0.0), state)

    def integrate(self, x0, t_end, dt):
        from torchdiffeq import odeint
        n_steps = int(t_end / dt) + 1
        t = torch.linspace(0.0, t_end, n_steps)
        return odeint(self.func, x0, t, method="dopri5")


# ---------------------------------------------------------------------------
# Training: Multiple Shooting
# ---------------------------------------------------------------------------

def integrate_window(model, x0_batch, n_steps, dt):
    """Integrate from batched initial conditions for n_steps."""
    state = x0_batch  # (n_windows, 2)
    trajectory = [state]
    for _ in range(n_steps):
        state = rk4_step(model.rhs, state, dt)
        trajectory.append(state)
    return torch.stack(trajectory, dim=1)  # (n_windows, n_steps+1, 2)


def compute_test_loss(model, x0, t_end, dt, traj_clean):
    """Integrate full trajectory from x0 and compute MSE vs clean ground truth."""
    with torch.no_grad():
        try:
            pred = model.integrate(x0, t_end, dt)
            n = min(pred.shape[0], traj_clean.shape[0])
            mse = F.mse_loss(pred[:n], traj_clean[:n]).item()
            if not math.isfinite(mse):
                mse = float("inf")
        except Exception:
            mse = float("inf")
    return mse


def train_model(model, true_traj, x0, t_end, dt, epochs, lr,
                traj_clean=None,
                window_size: int = 25, n_windows: int = 8,
                print_every: int = 500):
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    losses = []
    test_losses = []
    test_epochs = []
    total_steps = true_traj.shape[0] - 1
    eval_interval = max(1, epochs // 100)

    for epoch in range(epochs):
        starts = torch.randint(0, max(1, total_steps - window_size), (n_windows,))
        x0_batch = torch.stack([true_traj[s].detach() for s in starts])
        true_windows = torch.stack([
            true_traj[s:s + window_size + 1] for s in starts
        ])  # (n_windows, window_size+1, 2)

        pred_windows = integrate_window(model, x0_batch, window_size, dt)
        n = min(pred_windows.shape[1], true_windows.shape[1])
        loss = F.mse_loss(pred_windows[:, :n], true_windows[:, :n])

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

        # Periodic test loss evaluation
        if traj_clean is not None and (epoch % eval_interval == 0 or epoch == epochs - 1):
            test_mse = compute_test_loss(model, x0, t_end, dt, traj_clean)
            test_losses.append(test_mse)
            test_epochs.append(epoch)

        if (epoch + 1) % print_every == 0 or epoch == 0:
            msg = f"    Epoch {epoch+1:>5d}: loss = {loss.item():.6f}"
            if hasattr(model, "learned_params"):
                lp = model.learned_params()
                msg += f"  params: {', '.join(f'{k}={v:.4f}' for k, v in lp.items())}"
            print(msg)

    return losses, test_losses, test_epochs


def train_neural_ode(model, true_traj, x0, t_end, dt, epochs, lr,
                     traj_clean=None,
                     window_size: int = 25, n_windows: int = 8,
                     print_every: int = 500):
    """Train Neural ODE using torchdiffeq odeint for forward integration."""
    from torchdiffeq import odeint
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    losses = []
    test_losses = []
    test_epochs = []
    total_steps = true_traj.shape[0] - 1
    eval_interval = max(1, epochs // 100)

    for epoch in range(epochs):
        starts = torch.randint(0, max(1, total_steps - window_size), (n_windows,))
        total_loss = torch.tensor(0.0)

        for s in starts:
            t_window = torch.linspace(0.0, window_size * dt, window_size + 1)
            x0_w = true_traj[s].detach()
            true_w = true_traj[s:s + window_size + 1]
            pred_w = odeint(model.func, x0_w, t_window, method="dopri5")
            n = min(pred_w.shape[0], true_w.shape[0])
            total_loss = total_loss + F.mse_loss(pred_w[:n], true_w[:n])

        loss = total_loss / n_windows
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

        # Periodic test loss evaluation
        if traj_clean is not None and (epoch % eval_interval == 0 or epoch == epochs - 1):
            test_mse = compute_test_loss(model, x0, t_end, dt, traj_clean)
            test_losses.append(test_mse)
            test_epochs.append(epoch)

        if (epoch + 1) % print_every == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:>5d}: loss = {loss.item():.6f}")

    return losses, test_losses, test_epochs


def train_pinn(model, true_traj, t_train, t_end, epochs, lr,
               traj_clean=None, x0=None, dt=None,
               lambda_phys: float = 0.1, n_collocation: int = 200,
               print_every: int = 500):
    """Train PINN with data loss + physics residual loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    losses = []
    test_losses = []
    test_epochs = []
    eval_interval = max(1, epochs // 100)

    for epoch in range(epochs):
        # Data loss: PINN output at observation times vs true trajectory
        pred = model.forward(t_train)
        data_loss = F.mse_loss(pred, true_traj)

        # Physics loss: ODE residual at random collocation points
        t_coll = torch.rand(n_collocation) * t_end
        phys_loss = model.physics_loss(t_coll)

        loss = data_loss + lambda_phys * phys_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

        # Periodic test loss evaluation
        if traj_clean is not None and x0 is not None and dt is not None and (epoch % eval_interval == 0 or epoch == epochs - 1):
            test_mse = compute_test_loss(model, x0, t_end, dt, traj_clean)
            test_losses.append(test_mse)
            test_epochs.append(epoch)

        if (epoch + 1) % print_every == 0 or epoch == 0:
            msg = (f"    Epoch {epoch+1:>5d}: loss = {loss.item():.6f} "
                   f"(data={data_loss.item():.6f}, phys={phys_loss.item():.6f})")
            if hasattr(model, "learned_params"):
                lp = model.learned_params()
                msg += f"  params: {', '.join(f'{k}={v:.4f}' for k, v in lp.items())}"
            print(msg)

    return losses, test_losses, test_epochs


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@dataclass
class ODEResult:
    name: str
    n_params: int
    losses: list[float] = field(default_factory=list)
    test_losses: list[float] = field(default_factory=list)
    test_epochs: list[int] = field(default_factory=list)
    traj_mse_in: float = 0.0
    traj_mse_2x: float = 0.0
    traj_mse_5x: float = 0.0
    learned_params: dict[str, float] = field(default_factory=dict)
    param_errors: dict[str, float] = field(default_factory=dict)
    pred_traj_in: torch.Tensor | None = None
    pred_traj_2x: torch.Tensor | None = None
    pred_traj_5x: torch.Tensor | None = None


def evaluate_model(model, x0, dt, t_end_train, true_params):
    result = ODEResult(
        name=type(model).__name__,
        n_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    with torch.no_grad():
        for mult, attr_mse, attr_traj in [
            (1.0, "traj_mse_in", "pred_traj_in"),
            (2.0, "traj_mse_2x", "pred_traj_2x"),
            (5.0, "traj_mse_5x", "pred_traj_5x"),
        ]:
            t_end = t_end_train * mult
            _, traj_true = generate_trajectory(true_params, x0, t_end, dt)
            try:
                pred_traj = model.integrate(x0, t_end, dt)
                n = min(pred_traj.shape[0], traj_true.shape[0])
                mse = F.mse_loss(pred_traj[:n], traj_true[:n]).item()
                if not math.isfinite(mse):
                    mse = float("inf")
            except Exception:
                mse = float("inf")
                pred_traj = None
            setattr(result, attr_mse, mse)
            if pred_traj is not None:
                setattr(result, attr_traj, pred_traj.detach().clone())

    if hasattr(model, "learned_params"):
        result.learned_params = model.learned_params()
        for k, v in result.learned_params.items():
            if k in true_params and abs(true_params[k]) > 1e-8:
                result.param_errors[k] = abs(v - true_params[k]) / abs(true_params[k])

    return result


# ---------------------------------------------------------------------------
# Noise robustness
# ---------------------------------------------------------------------------

def noise_robustness(true_params, x0, t_end, dt, epochs, lr, noise_levels, seed=42):
    results = {}
    for noise in noise_levels:
        print(f"\n  Noise level: {noise*100:.0f}%")
        torch.manual_seed(seed)
        _, traj_clean = generate_trajectory(true_params, x0, t_end, dt)
        traj_noisy = add_noise(traj_clean, noise, seed=seed)

        m_prey, m_pred = compile_lv_rhs()
        model = LVKnownStructure(m_prey, m_pred)
        train_model(model, traj_noisy, x0, t_end, dt, epochs, lr,
                    traj_clean=traj_clean, print_every=epochs)

        lp = model.learned_params()
        errors = {}
        for k, v in lp.items():
            errors[k] = abs(v - true_params[k]) / abs(true_params[k])
        results[noise] = {"learned": lp, "errors": errors}
        max_err = max(errors.values())
        print(f"    Max param error: {max_err:.4%}")

    return results


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize(true_traj_in, true_traj_2x, true_traj_5x,
              t_in, t_2x, t_5x, results: list[ODEResult],
              noise_results: dict | None, save_path: str):
    short_names = {
        "LVKnownStructure": "Compiled",
        "LVHybrid": "Hybrid",
        "LVMLP": "MLP",
        "NeuralODE": "NeuralODE",
        "LVHandCoded": "Hand-coded",
        "LVPINN": "PINN",
    }
    colors = {"LVKnownStructure": "#2196F3", "LVHybrid": "#4CAF50",
              "LVMLP": "#FF9800", "NeuralODE": "#E91E63",
              "LVHandCoded": "#9C27B0", "LVPINN": "#795548"}

    def sn(name):
        return short_names.get(name, name)

    has_noise = noise_results and len(noise_results) > 0
    fig = plt.figure(figsize=(20, 20))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)

    # --- Panel 1: In-distribution trajectories (prey) ---
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t_in.numpy(), true_traj_in[:, 0].numpy(), "k-", linewidth=2, label="True prey")
    for r in results:
        if r.pred_traj_in is not None:
            c = colors.get(r.name, "gray")
            n = min(len(t_in), r.pred_traj_in.shape[0])
            ax.plot(t_in[:n].numpy(), r.pred_traj_in[:n, 0].numpy(),
                    "--", color=c, linewidth=1.5, label=sn(r.name))
    ax.set_xlabel("Time")
    ax.set_ylabel("Population")
    ax.set_title("In-Distribution: Prey")
    ax.legend(fontsize=9, loc="upper center")
    ax.grid(True, alpha=0.3)

    # --- Panel 2: In-distribution trajectories (predator) ---
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(t_in.numpy(), true_traj_in[:, 1].numpy(), "k-", linewidth=2, label="True predator")
    for r in results:
        if r.pred_traj_in is not None:
            c = colors.get(r.name, "gray")
            n = min(len(t_in), r.pred_traj_in.shape[0])
            ax.plot(t_in[:n].numpy(), r.pred_traj_in[:n, 1].numpy(),
                    "--", color=c, linewidth=1.5, label=sn(r.name))
    ax.set_xlabel("Time")
    ax.set_ylabel("Population")
    ax.set_title("In-Distribution: Predator")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel 3: 5x extrapolation (prey) ---
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(t_5x.numpy(), true_traj_5x[:, 0].numpy(), "k-", linewidth=2, label="True prey")
    ax.axvline(t_in[-1].item(), color="red", linestyle="--", alpha=0.8,
               linewidth=1.5, label="Training horizon")
    for r in results:
        if r.pred_traj_5x is not None:
            c = colors.get(r.name, "gray")
            n = min(len(t_5x), r.pred_traj_5x.shape[0])
            ax.plot(t_5x[:n].numpy(), r.pred_traj_5x[:n, 0].numpy(),
                    "--", color=c, linewidth=1.5, label=sn(r.name))
    ax.set_xlabel("Time")
    ax.set_ylabel("Population")
    ax.set_title("5x Extrapolation: Prey")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel 4: 5x extrapolation (predator) ---
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(t_5x.numpy(), true_traj_5x[:, 1].numpy(), "k-", linewidth=2, label="True predator")
    ax.axvline(t_in[-1].item(), color="red", linestyle="--", alpha=0.8,
               linewidth=1.5, label="Training horizon")
    for r in results:
        if r.pred_traj_5x is not None:
            c = colors.get(r.name, "gray")
            n = min(len(t_5x), r.pred_traj_5x.shape[0])
            ax.plot(t_5x[:n].numpy(), r.pred_traj_5x[:n, 1].numpy(),
                    "--", color=c, linewidth=1.5, label=sn(r.name))
    ax.set_xlabel("Time")
    ax.set_ylabel("Population")
    ax.set_title("5x Extrapolation: Predator")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel 5: Test loss ---
    ax = fig.add_subplot(gs[2, 0])
    for r in results:
        if r.test_losses and r.test_epochs:
            c = colors.get(r.name, "gray")
            ax.semilogy(r.test_epochs, r.test_losses, color=c, linewidth=1.5,
                        label=f"{sn(r.name)} ({r.n_params}p)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test MSE (log)")
    ax.set_title("Test Loss")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel 6: MSE comparison bar chart ---
    ax = fig.add_subplot(gs[2, 1])
    bar_names = [sn(r.name) for r in results]
    mse_in = [max(r.traj_mse_in, 1e-12) for r in results]
    mse_2x = [max(r.traj_mse_2x, 1e-12) for r in results]
    mse_5x = [max(r.traj_mse_5x, 1e-12) for r in results]
    x = np.arange(len(bar_names))
    w = 0.25
    ax.bar(x - w, mse_in, w, label="In-dist", color="#2196F3")
    ax.bar(x, mse_2x, w, label="2x extrap", color="#FF9800")
    ax.bar(x + w, mse_5x, w, label="5x extrap", color="#F44336")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(bar_names, fontsize=9)
    ax.set_ylabel("Trajectory MSE (log)")
    ax.set_title("MSE Comparison")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # --- Panel 7: Parameter recovery (compiled known-structure model) ---
    ax = fig.add_subplot(gs[3, 0])
    s1 = [r for r in results if r.name == "LVKnownStructure"]
    if s1:
        lp = s1[0].learned_params
        param_names = list(TRUE_PARAMS.keys())
        true_vals = [TRUE_PARAMS[k] for k in param_names]
        learned_vals = [lp.get(k, 0) for k in param_names]
        x = np.arange(len(param_names))
        ax.bar(x - 0.15, true_vals, 0.3, label="True", color="#4CAF50", alpha=0.8)
        ax.bar(x + 0.15, learned_vals, 0.3, label="Learned", color="#E91E63", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"${k}$" for k in param_names])
        ax.set_ylabel("Parameter value")
        ax.set_title("Parameter Recovery (Compiled Model)")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    # --- Panel 8: Noise robustness or extrapolation summary ---
    ax = fig.add_subplot(gs[3, 1])
    if has_noise:
        noise_levels = sorted(noise_results.keys())
        param_names = list(TRUE_PARAMS.keys())
        for pname in param_names:
            errs = [noise_results[n]["errors"].get(pname, 0) for n in noise_levels]
            ax.plot([n * 100 for n in noise_levels], errs, "o-", label=f"${pname}$")
        ax.set_xlabel("Noise level (%)")
        ax.set_ylabel("Relative parameter error")
        ax.set_title("Noise Robustness (Compiled Model)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")
    else:
        sorted_res = sorted(results, key=lambda r: r.traj_mse_5x)
        model_labels = [sn(r.name) for r in sorted_res]
        mse_vals = [r.traj_mse_5x for r in sorted_res]
        bar_colors = [colors.get(r.name, "gray") for r in sorted_res]
        y = np.arange(len(model_labels))
        ax.barh(y, mse_vals, color=bar_colors, alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels(model_labels, fontsize=10)
        ax.set_xscale("log")
        ax.set_xlabel("Trajectory MSE (log)")
        ax.set_title("5x Extrapolation MSE (lower is better)")
        ax.grid(True, alpha=0.3, axis="x")

    fig.suptitle("Lotka-Volterra: Structure-Aware ODE Learning",
                 fontsize=16, fontweight="bold", y=0.995)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to {save_path}")
    plt.close()


def visualize_phase_space(true_traj_5x, results: list[ODEResult], save_path: str):
    if len(results) == 0:
        return
    colors = {"LVKnownStructure": "#2196F3", "LVHybrid": "#4CAF50",
              "LVMLP": "#FF9800", "NeuralODE": "#E91E63",
              "LVHandCoded": "#9C27B0", "LVPINN": "#795548"}
    short_names = {
        "LVKnownStructure": "Compiled",
        "LVHybrid": "Hybrid",
        "LVMLP": "MLP",
        "NeuralODE": "NeuralODE",
        "LVHandCoded": "Hand-coded",
        "LVPINN": "PINN",
    }

    n_panels = len(results) + 1
    ncols = 4
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes_flat = axes.flatten()

    for extra_ax in axes_flat[n_panels:]:
        extra_ax.set_visible(False)

    ax = axes_flat[0]
    ax.plot(true_traj_5x[:, 0].numpy(), true_traj_5x[:, 1].numpy(), "k-", linewidth=1.5)
    ax.plot(true_traj_5x[0, 0].item(), true_traj_5x[0, 1].item(), "go", markersize=8)
    ax.set_xlabel("Prey", fontsize=11)
    ax.set_ylabel("Predator", fontsize=11)
    ax.set_title("Ground Truth", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    for i, r in enumerate(results):
        ax = axes_flat[i + 1]
        ax.plot(true_traj_5x[:, 0].numpy(), true_traj_5x[:, 1].numpy(),
                "k-", linewidth=0.5, alpha=0.3, label="True")
        if r.pred_traj_5x is not None:
            n = min(true_traj_5x.shape[0], r.pred_traj_5x.shape[0])
            c = colors.get(r.name, "gray")
            sn = short_names.get(r.name, r.name)
            ax.plot(r.pred_traj_5x[:n, 0].numpy(), r.pred_traj_5x[:n, 1].numpy(),
                    "-", color=c, linewidth=1.5, label=sn)
        ax.plot(true_traj_5x[0, 0].item(), true_traj_5x[0, 1].item(), "go", markersize=8)
        ax.set_xlabel("Prey", fontsize=11)
        if (i + 1) % ncols == 0:
            ax.set_ylabel("Predator", fontsize=11)
        sn = short_names.get(r.name, r.name)
        ax.set_title(f"{sn} ({r.n_params:,}p)", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle("Phase Space (5x extrapolation)", fontsize=16, fontweight="bold")
    plt.tight_layout()
    phase_path = save_path.replace(".png", "_phase.png")
    plt.savefig(phase_path, dpi=150, bbox_inches="tight")
    print(f"Phase space figure saved to {phase_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results: list[ODEResult], true_params: dict):
    print("\n" + "=" * 100)
    print("LOTKA-VOLTERRA ODE EXPERIMENT — RESULTS SUMMARY")
    print("=" * 100)

    header = f"{'Model':<22s} {'Params':>7s} {'MSE(in)':>12s} {'MSE(2x)':>12s} {'MSE(5x)':>12s}"
    print(header)
    print("-" * 70)
    for r in results:
        m5 = f"{r.traj_mse_5x:.6f}" if math.isfinite(r.traj_mse_5x) else "inf"
        m2 = f"{r.traj_mse_2x:.6f}" if math.isfinite(r.traj_mse_2x) else "inf"
        mi = f"{r.traj_mse_in:.6f}" if math.isfinite(r.traj_mse_in) else "inf"
        print(f"{r.name:<22s} {r.n_params:>7d} {mi:>12s} {m2:>12s} {m5:>12s}")

    print("\nPARAMETER RECOVERY:")
    for r in results:
        if r.learned_params:
            print(f"\n  {r.name}:")
            for k, v in r.learned_params.items():
                if k in true_params:
                    err = r.param_errors.get(k, float("inf"))
                    print(f"    {k}: true={true_params[k]:.4f}, learned={v:.4f}, "
                          f"error={err:.4%}")

    print("\nIMPROVEMENT RATIOS (Scenario 1 vs baselines):")
    s1 = [r for r in results if r.name == "LVKnownStructure"]
    baselines = [r for r in results if r.name in ("LVMLP", "NeuralODE")]
    if s1 and baselines:
        s1_r = s1[0]
        for bl in baselines:
            if math.isfinite(s1_r.traj_mse_in) and s1_r.traj_mse_in > 0:
                ratio_in = bl.traj_mse_in / s1_r.traj_mse_in
                ratio_2x = bl.traj_mse_2x / max(s1_r.traj_mse_2x, 1e-12)
                ratio_5x = bl.traj_mse_5x / max(s1_r.traj_mse_5x, 1e-12)
                print(f"  vs {bl.name}: "
                      f"{ratio_in:.1f}x (in), {ratio_2x:.1f}x (2x), {ratio_5x:.1f}x (5x)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Lotka-Volterra ODE experiment")
    parser.add_argument("--t-end", type=float, default=12.0,
                        help="Training time horizon")
    parser.add_argument("--dt", type=float, default=0.1,
                        help="Integration time step")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--noise", type=float, default=0.02,
                        help="Observation noise level")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-size", type=int, default=25,
                        help="Multiple shooting window size")
    parser.add_argument("--save-fig", default="examples/ode_lotka_volterra.png")
    parser.add_argument("--skip-neural-ode", action="store_true",
                        help="Skip Neural ODE baseline")
    parser.add_argument("--noise-robustness", action="store_true",
                        help="Run noise robustness analysis")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip training; load saved data and regenerate figures")
    args = parser.parse_args()

    # --- Plot-only mode: load saved data and regenerate figures ---
    if args.plot_only:
        data_path = args.save_fig.replace(".png", "_data.pt")
        print(f"Loading saved results from {data_path}")
        save_data = torch.load(data_path, weights_only=False)
        visualize(
            save_data["true_traj_in"], save_data["true_traj_2x"],
            save_data["true_traj_5x"], save_data["t_in"],
            save_data["t_2x"], save_data["t_5x"],
            save_data["results"], save_data["noise_results"],
            args.save_fig,
        )
        visualize_phase_space(
            save_data["true_traj_5x"], save_data["results"], args.save_fig,
        )
        return

    torch.manual_seed(args.seed)
    print("Lotka-Volterra ODE Experiment")
    print(f"  True params: {TRUE_PARAMS}")
    print(f"  Training horizon: [0, {args.t_end}]")
    print(f"  dt={args.dt}, epochs={args.epochs}, lr={args.lr}")
    print(f"  Noise level: {args.noise*100:.0f}%, window_size={args.window_size}")
    print()

    # --- Generate ground truth ---
    t_in, traj_true = generate_trajectory(TRUE_PARAMS, X0_DEFAULT, args.t_end, args.dt)
    t_2x, traj_true_2x = generate_trajectory(TRUE_PARAMS, X0_DEFAULT, args.t_end * 2, args.dt)
    t_5x, traj_true_5x = generate_trajectory(TRUE_PARAMS, X0_DEFAULT, args.t_end * 5, args.dt)

    traj_noisy = add_noise(traj_true, args.noise, seed=args.seed)
    print(f"  Trajectory: {traj_true.shape[0]} steps, x0={X0_DEFAULT.tolist()}")
    print(f"  Prey range: [{traj_true[:,0].min():.2f}, {traj_true[:,0].max():.2f}]")
    print(f"  Predator range: [{traj_true[:,1].min():.2f}, {traj_true[:,1].max():.2f}]")
    print()

    results = []

    # --- Scenario 1: Known structure, unknown parameters ---
    print("=" * 60)
    print("SCENARIO 1: Known structure, unknown parameters")
    print("=" * 60)
    m_prey, m_pred = compile_lv_rhs()
    model_s1 = LVKnownStructure(m_prey, m_pred)
    n_params_s1 = sum(p.numel() for p in model_s1.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params_s1}")
    print(f"  Initial params: {model_s1.learned_params()}")
    print()

    torch.manual_seed(args.seed)
    t0 = time.time()
    losses_s1, test_losses_s1, test_epochs_s1 = train_model(
        model_s1, traj_noisy, X0_DEFAULT, args.t_end, args.dt,
        args.epochs, args.lr, traj_clean=traj_true, window_size=args.window_size)
    dt_s1 = time.time() - t0
    print(f"  Training time: {dt_s1:.1f}s")
    result_s1 = evaluate_model(model_s1, X0_DEFAULT, args.dt, args.t_end, TRUE_PARAMS)
    result_s1.losses = losses_s1
    result_s1.test_losses = test_losses_s1
    result_s1.test_epochs = test_epochs_s1
    results.append(result_s1)

    # --- Scenario 2: Known interaction + learned growth/death ---
    print("\n" + "=" * 60)
    print("SCENARIO 2: Known interaction + learned growth/death")
    print("=" * 60)
    m_predation = compile_predation()
    model_s2 = LVHybrid(m_predation)
    n_params_s2 = sum(p.numel() for p in model_s2.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params_s2}")
    print()

    torch.manual_seed(args.seed)
    t0 = time.time()
    losses_s2, test_losses_s2, test_epochs_s2 = train_model(
        model_s2, traj_noisy, X0_DEFAULT, args.t_end, args.dt,
        args.epochs, args.lr * 0.3, traj_clean=traj_true, window_size=args.window_size)
    dt_s2 = time.time() - t0
    print(f"  Training time: {dt_s2:.1f}s")
    result_s2 = evaluate_model(model_s2, X0_DEFAULT, args.dt, args.t_end, TRUE_PARAMS)
    result_s2.name = "LVHybrid"
    result_s2.losses = losses_s2
    result_s2.test_losses = test_losses_s2
    result_s2.test_epochs = test_epochs_s2
    results.append(result_s2)

    # --- Baseline: Pure MLP ---
    print("\n" + "=" * 60)
    print("BASELINE: Pure MLP RHS")
    print("=" * 60)
    model_mlp = LVMLP(hidden=64, layers=3)
    n_params_mlp = sum(p.numel() for p in model_mlp.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params_mlp}")
    print()

    torch.manual_seed(args.seed)
    t0 = time.time()
    losses_mlp, test_losses_mlp, test_epochs_mlp = train_model(
        model_mlp, traj_noisy, X0_DEFAULT, args.t_end, args.dt,
        args.epochs, args.lr, traj_clean=traj_true, window_size=args.window_size)
    dt_mlp = time.time() - t0
    print(f"  Training time: {dt_mlp:.1f}s")
    result_mlp = evaluate_model(model_mlp, X0_DEFAULT, args.dt, args.t_end, TRUE_PARAMS)
    result_mlp.name = "LVMLP"
    result_mlp.losses = losses_mlp
    result_mlp.test_losses = test_losses_mlp
    result_mlp.test_epochs = test_epochs_mlp
    results.append(result_mlp)

    # --- Baseline: Neural ODE ---
    if not args.skip_neural_ode:
        print("\n" + "=" * 60)
        print("BASELINE: Neural ODE (torchdiffeq)")
        print("=" * 60)
        try:
            model_node = NeuralODE(hidden=64, layers=3)
            n_params_node = sum(p.numel() for p in model_node.parameters() if p.requires_grad)
            print(f"  Trainable parameters: {n_params_node}")
            print()

            torch.manual_seed(args.seed)
            t0 = time.time()
            losses_node, test_losses_node, test_epochs_node = train_neural_ode(
                model_node, traj_noisy, X0_DEFAULT, args.t_end,
                args.dt, args.epochs, args.lr,
                traj_clean=traj_true, window_size=args.window_size)
            dt_node = time.time() - t0
            print(f"  Training time: {dt_node:.1f}s")
            result_node = evaluate_model(model_node, X0_DEFAULT, args.dt, args.t_end,
                                         TRUE_PARAMS)
            result_node.name = "NeuralODE"
            result_node.losses = losses_node
            result_node.test_losses = test_losses_node
            result_node.test_epochs = test_epochs_node
            results.append(result_node)
        except Exception as e:
            print(f"  Neural ODE failed: {e}")

    # --- Baseline: Hand-coded PyTorch ---
    print("\n" + "=" * 60)
    print("BASELINE: Hand-coded PyTorch LV equations")
    print("=" * 60)
    model_hc = LVHandCoded()
    n_params_hc = sum(p.numel() for p in model_hc.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params_hc}")
    print(f"  Initial params: {model_hc.learned_params()}")
    print()

    torch.manual_seed(args.seed)
    t0 = time.time()
    losses_hc, test_losses_hc, test_epochs_hc = train_model(
        model_hc, traj_noisy, X0_DEFAULT, args.t_end, args.dt,
        args.epochs, args.lr, traj_clean=traj_true, window_size=args.window_size)
    dt_hc = time.time() - t0
    print(f"  Training time: {dt_hc:.1f}s")
    result_hc = evaluate_model(model_hc, X0_DEFAULT, args.dt, args.t_end, TRUE_PARAMS)
    result_hc.name = "LVHandCoded"
    result_hc.losses = losses_hc
    result_hc.test_losses = test_losses_hc
    result_hc.test_epochs = test_epochs_hc
    results.append(result_hc)

    # --- Baseline: PINN ---
    print("\n" + "=" * 60)
    print("BASELINE: Physics-Informed Neural Network (PINN)")
    print("=" * 60)
    model_pinn = LVPINN(hidden=64, layers=3)
    n_params_pinn = sum(p.numel() for p in model_pinn.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params_pinn}")
    print()

    torch.manual_seed(args.seed)
    t0 = time.time()
    losses_pinn, test_losses_pinn, test_epochs_pinn = train_pinn(
        model_pinn, traj_noisy, t_in, args.t_end,
        args.epochs, args.lr, traj_clean=traj_true, x0=X0_DEFAULT, dt=args.dt)
    dt_pinn = time.time() - t0
    print(f"  Training time: {dt_pinn:.1f}s")
    result_pinn = evaluate_model(model_pinn, X0_DEFAULT, args.dt, args.t_end, TRUE_PARAMS)
    result_pinn.name = "LVPINN"
    result_pinn.losses = losses_pinn
    result_pinn.test_losses = test_losses_pinn
    result_pinn.test_epochs = test_epochs_pinn
    results.append(result_pinn)

    # --- Summary ---
    print_summary(results, TRUE_PARAMS)

    # --- Noise robustness ---
    noise_results = None
    if args.noise_robustness:
        print("\n" + "=" * 60)
        print("NOISE ROBUSTNESS ANALYSIS")
        print("=" * 60)
        noise_results = noise_robustness(
            TRUE_PARAMS, X0_DEFAULT, args.t_end, args.dt,
            args.epochs, args.lr,
            noise_levels=[0.0, 0.01, 0.02, 0.05, 0.10],
            seed=args.seed,
        )

    # --- Save data for plot-only regeneration ---
    data_path = args.save_fig.replace(".png", "_data.pt")
    save_data = {
        "true_traj_in": traj_true,
        "true_traj_2x": traj_true_2x,
        "true_traj_5x": traj_true_5x,
        "t_in": t_in,
        "t_2x": t_2x,
        "t_5x": t_5x,
        "results": results,
        "noise_results": noise_results,
    }
    torch.save(save_data, data_path)
    print(f"\nResults saved to {data_path}")

    # --- Visualization ---
    visualize(traj_true, traj_true_2x, traj_true_5x,
              t_in, t_2x, t_5x, results, noise_results, args.save_fig)
    visualize_phase_space(traj_true_5x, results, args.save_fig)


if __name__ == "__main__":
    main()

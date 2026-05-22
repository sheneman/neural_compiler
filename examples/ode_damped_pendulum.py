############################################################
#
# Neural Compiler: Compiling scheme into composable and
#                  differentiable neural network representations
#
# ode_damped_pendulum.py: Compiled transcendental gravity (sin) with learned damping and forcing dynamics
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Partially-known ODE systems: Damped pendulum with external forcing.

The damped pendulum demonstrates compiling a transcendental known term (gravity
with sin) while learning unknown dynamics (damping + forcing) with an MLP.

Full system: theta'' = -(g/L)*sin(theta) - b*theta' + F_amp*sin(F_freq*t)

Scenarios:
  1. Known full structure (F=0), learn g/L and b via compiled subgraph
  2. Compile gravity term -(g/L)*sin(theta), learn damping + forcing with MLP

The compiled gravity term uses the sin operation (v0.6.0) — this is the first
ODE experiment requiring transcendental compilation.

Baselines:
  - Pure MLP: learns full second-order dynamics from scratch
  - Neural ODE (torchdiffeq): continuous-depth MLP, adaptive step solver
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
    "g_over_L": 9.81,       # g/L for L=1m
    "b": 0.3,               # damping coefficient
    "F_amp": 0.5,           # forcing amplitude
    "F_freq": 2.0 / 3.0,   # forcing frequency (sub-resonant)
}

X0_DEFAULT = torch.tensor([1.0, 0.0])  # (theta, theta_dot) initial condition


# ---------------------------------------------------------------------------
# Ground truth ODE RHS
# ---------------------------------------------------------------------------

def pendulum_rhs(state: torch.Tensor, t: float | torch.Tensor,
                 params: dict) -> torch.Tensor:
    theta, theta_dot = state[..., 0], state[..., 1]
    gravity = -params["g_over_L"] * torch.sin(theta)
    damping = -params["b"] * theta_dot
    if isinstance(t, torch.Tensor):
        forcing = params["F_amp"] * torch.sin(params["F_freq"] * t)
    else:
        forcing = params["F_amp"] * math.sin(params["F_freq"] * t)
    theta_ddot = gravity + damping + forcing
    return torch.stack([theta_dot, theta_ddot], dim=-1)


def rk4_step_t(rhs_fn, state, t, dt):
    k1 = rhs_fn(state, t)
    k2 = rhs_fn(state + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = rhs_fn(state + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = rhs_fn(state + dt * k3, t + dt)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def generate_trajectory(params: dict, x0: torch.Tensor,
                        t_end: float, dt: float = 0.05) -> tuple[torch.Tensor, torch.Tensor]:
    def rhs(state, t):
        return pendulum_rhs(state, t, params)

    steps = int(t_end / dt)
    trajectory = [x0]
    state = x0
    t_val = 0.0
    for _ in range(steps):
        state = rk4_step_t(rhs, state, t_val, dt)
        t_val += dt
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

def compile_gravity_term():
    scheme = "(* neg_g_over_L (sin theta))"
    inputs = {"neg_g_over_L": None, "theta": None}
    g = compile_scheme(scheme, inputs=inputs)
    m = DirectModule(g)
    for p in m.parameters():
        p.requires_grad = False
    m.eval()
    print(f"  Compiled gravity term: {len(g.nodes)} nodes, depth={g.depth()}")
    return m


def compile_full_pendulum_rhs():
    scheme = "(+ (* neg_g_over_L (sin theta)) (* neg_b theta_dot))"
    inputs = {"neg_g_over_L": None, "neg_b": None, "theta": None, "theta_dot": None}
    g = compile_scheme(scheme, inputs=inputs)
    m = DirectModule(g)
    for p in m.parameters():
        p.requires_grad = False
    m.eval()
    print(f"  Compiled full pendulum RHS: {len(g.nodes)} nodes, depth={g.depth()}")
    return m


# ---------------------------------------------------------------------------
# Model: Scenario 1 — Known structure, learn g/L and b (no forcing)
# ---------------------------------------------------------------------------

class PendulumKnownStructure(nn.Module):
    """Full pendulum RHS compiled; learn g/L and b. Assumes F=0."""

    def __init__(self, m_rhs: DirectModule):
        super().__init__()
        self.m_rhs = m_rhs
        self.neg_g_over_L = nn.Parameter(torch.tensor(-5.0))
        self.neg_b = nn.Parameter(torch.tensor(-0.5))

    def rhs(self, state, t=None):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        theta, theta_dot = state[:, 0], state[:, 1]
        batch = theta.shape[0]

        inputs = {
            "neg_g_over_L": self.neg_g_over_L.expand(batch),
            "neg_b": self.neg_b.expand(batch),
            "theta": theta,
            "theta_dot": theta_dot,
        }
        theta_ddot = self.m_rhs.forward_batch(inputs)
        return torch.stack([theta_dot, theta_ddot], dim=-1)

    def integrate(self, x0, t_end, dt):
        steps = int(t_end / dt)
        trajectory = [x0.unsqueeze(0)]
        state = x0.unsqueeze(0)
        t_val = 0.0
        for _ in range(steps):
            state = rk4_step_t(self.rhs, state, t_val, dt)
            t_val += dt
            trajectory.append(state)
        return torch.cat(trajectory, dim=0).squeeze(1)

    def learned_params(self):
        return {
            "g_over_L": -self.neg_g_over_L.item(),
            "b": -self.neg_b.item(),
        }


# ---------------------------------------------------------------------------
# Model: Scenario 2 — Compiled gravity + learned damping/forcing
# ---------------------------------------------------------------------------

class PendulumHybrid(nn.Module):
    """Compiled gravity term + MLP for damping + forcing."""

    def __init__(self, m_gravity: DirectModule, hidden: int = 32):
        super().__init__()
        self.m_gravity = m_gravity
        self.neg_g_over_L = nn.Parameter(torch.tensor(-5.0))
        self.correction_net = nn.Sequential(
            nn.Linear(3, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.correction_net[-1].weight)
        nn.init.zeros_(self.correction_net[-1].bias)

    def rhs(self, state, t=None):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        theta, theta_dot = state[:, 0], state[:, 1]
        batch = theta.shape[0]

        gravity_inputs = {
            "neg_g_over_L": self.neg_g_over_L.expand(batch),
            "theta": theta,
        }
        gravity = self.m_gravity.forward_batch(gravity_inputs)

        if t is None:
            t = 0.0
        t_val = torch.full((batch,), t if isinstance(t, (int, float)) else t.item())
        correction_input = torch.stack([theta, theta_dot, t_val], dim=-1)
        correction = self.correction_net(correction_input).squeeze(-1)

        theta_ddot = gravity + correction
        return torch.stack([theta_dot, theta_ddot], dim=-1)

    def integrate(self, x0, t_end, dt):
        steps = int(t_end / dt)
        trajectory = [x0.unsqueeze(0)]
        state = x0.unsqueeze(0)
        t_val = 0.0
        for _ in range(steps):
            state = rk4_step_t(self.rhs, state, t_val, dt)
            t_val += dt
            trajectory.append(state)
        return torch.cat(trajectory, dim=0).squeeze(1)

    def learned_params(self):
        return {"g_over_L": -self.neg_g_over_L.item()}


# ---------------------------------------------------------------------------
# Baseline: Pure MLP
# ---------------------------------------------------------------------------

class PendulumMLP(nn.Module):
    def __init__(self, hidden: int = 64, layers: int = 3):
        super().__init__()
        modules = [nn.Linear(3, hidden), nn.Tanh()]
        for _ in range(layers - 1):
            modules.extend([nn.Linear(hidden, hidden), nn.Tanh()])
        modules.append(nn.Linear(hidden, 2))
        self.net = nn.Sequential(*modules)

    def rhs(self, state, t=None):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        theta, theta_dot = state[:, 0], state[:, 1]
        batch = theta.shape[0]
        if t is None:
            t = 0.0
        t_val = torch.full((batch,), t if isinstance(t, (int, float)) else t.item())
        x = torch.stack([theta, theta_dot, t_val], dim=-1)
        return self.net(x)

    def integrate(self, x0, t_end, dt):
        steps = int(t_end / dt)
        trajectory = [x0.unsqueeze(0)]
        state = x0.unsqueeze(0)
        t_val = 0.0
        for _ in range(steps):
            state = rk4_step_t(self.rhs, state, t_val, dt)
            t_val += dt
            trajectory.append(state)
        return torch.cat(trajectory, dim=0).squeeze(1)


# ---------------------------------------------------------------------------
# Baseline: Neural ODE
# ---------------------------------------------------------------------------

class PendulumHandCoded(nn.Module):
    """Hand-coded PyTorch: same pendulum equations, no compiler."""

    def __init__(self):
        super().__init__()
        self.neg_g_over_L = nn.Parameter(torch.tensor(-5.0))
        self.neg_b = nn.Parameter(torch.tensor(-0.5))

    def rhs(self, state, t=None):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        theta, theta_dot = state[:, 0], state[:, 1]
        theta_ddot = self.neg_g_over_L * torch.sin(theta) + self.neg_b * theta_dot
        return torch.stack([theta_dot, theta_ddot], dim=-1)

    def integrate(self, x0, t_end, dt):
        steps = int(t_end / dt)
        trajectory = [x0.unsqueeze(0)]
        state = x0.unsqueeze(0)
        t_val = 0.0
        for _ in range(steps):
            state = rk4_step_t(self.rhs, state, t_val, dt)
            t_val += dt
            trajectory.append(state)
        return torch.cat(trajectory, dim=0).squeeze(1)

    def learned_params(self):
        return {"g_over_L": -self.neg_g_over_L.item(), "b": -self.neg_b.item()}


# ---------------------------------------------------------------------------
# Baseline: PINN
# ---------------------------------------------------------------------------

class PendulumPINN(nn.Module):
    """PINN: MLP approximates trajectory, ODE residual as soft loss."""

    def __init__(self, hidden: int = 64, layers: int = 3):
        super().__init__()
        mods = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(layers - 1):
            mods.extend([nn.Linear(hidden, hidden), nn.Tanh()])
        mods.append(nn.Linear(hidden, 2))
        self.net = nn.Sequential(*mods)
        # Trainable physics parameters
        self.neg_g_over_L = nn.Parameter(torch.tensor(-5.0))
        self.neg_b = nn.Parameter(torch.tensor(-0.5))

    def forward(self, t):
        """t: (N,) -> (N, 2) predicted [theta, theta_dot]."""
        return self.net(t.unsqueeze(-1) if t.dim() == 1 else t)

    def physics_loss(self, t_collocation):
        """ODE residual loss at collocation points."""
        t = t_collocation.unsqueeze(-1).requires_grad_(True)
        u = self.net(t)  # (N, 2)
        theta, theta_dot = u[:, 0:1], u[:, 1:2]

        # d(theta)/dt should equal theta_dot (kinematic constraint)
        dtheta_dt = torch.autograd.grad(
            theta, t, grad_outputs=torch.ones_like(theta), create_graph=True
        )[0]
        # d(theta_dot)/dt should equal theta_ddot
        dthetadot_dt = torch.autograd.grad(
            theta_dot, t, grad_outputs=torch.ones_like(theta_dot), create_graph=True
        )[0]

        # Physics: theta_ddot = neg_g_over_L * sin(theta) + neg_b * theta_dot
        rhs_theta = theta_dot
        rhs_thetadot = self.neg_g_over_L * torch.sin(theta) + self.neg_b * theta_dot

        residual = ((dtheta_dt - rhs_theta).pow(2).mean()
                     + (dthetadot_dt - rhs_thetadot).pow(2).mean())
        return residual

    def integrate(self, x0, t_end, dt):
        """Evaluate the trained PINN at discrete time points."""
        n_steps = int(t_end / dt) + 1
        t = torch.linspace(0.0, t_end, n_steps)
        with torch.no_grad():
            return self.forward(t)

    def learned_params(self):
        return {"g_over_L": -self.neg_g_over_L.item(), "b": -self.neg_b.item()}


# ---------------------------------------------------------------------------
# Baseline: Neural ODE
# ---------------------------------------------------------------------------

class PendulumNODEFunc(nn.Module):
    def __init__(self, hidden: int = 64, layers: int = 3):
        super().__init__()
        modules = [nn.Linear(3, hidden), nn.Tanh()]
        for _ in range(layers - 1):
            modules.extend([nn.Linear(hidden, hidden), nn.Tanh()])
        modules.append(nn.Linear(hidden, 2))
        self.net = nn.Sequential(*modules)

    def forward(self, t, y):
        if y.dim() == 1:
            y = y.unsqueeze(0)
        batch = y.shape[0]
        t_val = torch.full((batch, 1), t.item() if isinstance(t, torch.Tensor) else t)
        x = torch.cat([y, t_val], dim=-1)
        out = self.net(x)
        return out.squeeze(0) if y.shape[0] == 1 else out


class PendulumNeuralODE(nn.Module):
    def __init__(self, hidden: int = 64, layers: int = 3):
        super().__init__()
        self.func = PendulumNODEFunc(hidden, layers)

    def rhs(self, state, t=None):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return self.func(torch.tensor(0.0), state)

    def integrate(self, x0, t_end, dt):
        from torchdiffeq import odeint
        n_steps = int(t_end / dt) + 1
        t = torch.linspace(0.0, t_end, n_steps)
        return odeint(self.func, x0, t, method="dopri5")


# ---------------------------------------------------------------------------
# Training: Multiple Shooting (time-aware)
# ---------------------------------------------------------------------------

def integrate_window_t(model, x0_batch, t_starts, n_steps, dt):
    """Integrate batched windows, each starting at different t values."""
    state = x0_batch  # (n_windows, 2)
    trajectory = [state]
    for step in range(n_steps):
        t_batch = t_starts + step * dt
        # All windows share the same model call
        derivs = []
        for i in range(state.shape[0]):
            d = model.rhs(state[i:i+1], t_batch[i].item())
            derivs.append(d)
        deriv = torch.cat(derivs, dim=0)

        k1 = deriv
        # For RK4 sub-steps, approximate with the same t
        s2 = state + 0.5 * dt * k1
        derivs2 = []
        for i in range(s2.shape[0]):
            d = model.rhs(s2[i:i+1], (t_batch[i] + 0.5 * dt).item())
            derivs2.append(d)
        k2 = torch.cat(derivs2, dim=0)

        s3 = state + 0.5 * dt * k2
        derivs3 = []
        for i in range(s3.shape[0]):
            d = model.rhs(s3[i:i+1], (t_batch[i] + 0.5 * dt).item())
            derivs3.append(d)
        k3 = torch.cat(derivs3, dim=0)

        s4 = state + dt * k3
        derivs4 = []
        for i in range(s4.shape[0]):
            d = model.rhs(s4[i:i+1], (t_batch[i] + dt).item())
            derivs4.append(d)
        k4 = torch.cat(derivs4, dim=0)

        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
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
                window_size: int = 25, n_windows: int = 6,
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
        ])

        t_starts = starts.float() * dt
        pred_windows = integrate_window_t(model, x0_batch, t_starts, window_size, dt)
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
                     window_size: int = 25, n_windows: int = 6,
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
            t_start = s.item() * dt
            t_window = torch.linspace(t_start, t_start + window_size * dt, window_size + 1)
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
class PendulumResult:
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
    pred_traj_5x: torch.Tensor | None = None


def evaluate_model(model, x0, dt, t_end_train, true_params):
    result = PendulumResult(
        name=type(model).__name__,
        n_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    with torch.no_grad():
        for mult, attr_mse, attr_traj in [
            (1.0, "traj_mse_in", "pred_traj_in"),
            (2.0, "traj_mse_2x", None),
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
            if attr_traj and pred_traj is not None:
                setattr(result, attr_traj, pred_traj.detach().clone())

    if hasattr(model, "learned_params"):
        result.learned_params = model.learned_params()
        for k, v in result.learned_params.items():
            if k in true_params and abs(true_params[k]) > 1e-8:
                result.param_errors[k] = abs(v - true_params[k]) / abs(true_params[k])

    return result


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize(true_traj_in, true_traj_5x, t_in, t_5x,
              results: list[PendulumResult], save_path: str):
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
    colors = {"PendulumKnownStructure": "#2196F3", "PendulumHybrid": "#4CAF50",
              "PendulumMLP": "#FF9800", "PendulumNeuralODE": "#E91E63",
              "PendulumHandCoded": "#9C27B0", "PendulumPINN": "#795548"}
    short_names = {
        "PendulumHybrid": "Hybrid",
        "PendulumMLP": "MLP",
        "PendulumNeuralODE": "NeuralODE",
        "PendulumKnownStructure": "Compiled",
        "PendulumHandCoded": "Hand-coded",
        "PendulumPINN": "PINN",
    }

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t_in.numpy(), true_traj_in[:, 0].numpy(), "k-", linewidth=2, label="True")
    for r in results:
        if r.pred_traj_in is not None:
            c = colors.get(r.name, "gray")
            sn = short_names.get(r.name, r.name)
            n = min(len(t_in), r.pred_traj_in.shape[0])
            ax.plot(t_in[:n].numpy(), r.pred_traj_in[:n, 0].numpy(),
                    "--", color=c, linewidth=1.5, label=sn)
    ax.set_xlabel("Time")
    ax.set_ylabel("theta (rad)")
    ax.set_title("In-Distribution: Angle")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(t_in.numpy(), true_traj_in[:, 1].numpy(), "k-", linewidth=2, label="True")
    for r in results:
        if r.pred_traj_in is not None:
            c = colors.get(r.name, "gray")
            sn = short_names.get(r.name, r.name)
            n = min(len(t_in), r.pred_traj_in.shape[0])
            ax.plot(t_in[:n].numpy(), r.pred_traj_in[:n, 1].numpy(),
                    "--", color=c, linewidth=1.5, label=sn)
    ax.set_xlabel("Time")
    ax.set_ylabel("theta_dot (rad/s)")
    ax.set_title("In-Distribution: Angular Velocity")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(t_5x.numpy(), true_traj_5x[:, 0].numpy(), "k-", linewidth=2, label="True")
    ax.axvline(t_in[-1].item(), color="gray", linestyle=":", alpha=0.5, label="Training horizon")
    for r in results:
        if r.pred_traj_5x is not None:
            c = colors.get(r.name, "gray")
            sn = short_names.get(r.name, r.name)
            n = min(len(t_5x), r.pred_traj_5x.shape[0])
            ax.plot(t_5x[:n].numpy(), r.pred_traj_5x[:n, 0].numpy(),
                    "--", color=c, linewidth=1.5, label=sn)
    ax.set_xlabel("Time")
    ax.set_ylabel("theta (rad)")
    ax.set_title("5x Extrapolation: Angle")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 1])
    names = [short_names.get(r.name, r.name) for r in results]
    mse_in = [max(r.traj_mse_in, 1e-12) for r in results]
    mse_2x = [max(r.traj_mse_2x, 1e-12) for r in results]
    mse_5x = [max(r.traj_mse_5x, 1e-12) for r in results]
    x = np.arange(len(names))
    w = 0.25
    ax.bar(x - w, mse_in, w, label="In-dist", color="#2196F3")
    ax.bar(x, mse_2x, w, label="2x extrap", color="#FF9800")
    ax.bar(x + w, mse_5x, w, label="5x extrap", color="#F44336")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Trajectory MSE (log)")
    ax.set_title("MSE Comparison")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    ax = fig.add_subplot(gs[2, 0])
    for r in results:
        if r.test_losses and r.test_epochs:
            c = colors.get(r.name, "gray")
            sn = short_names.get(r.name, r.name)
            ax.semilogy(r.test_epochs, r.test_losses, color=c, linewidth=1.5,
                        label=f"{sn} ({r.n_params}p)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test MSE (log)")
    ax.set_title("Test Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[2, 1])
    param_data = []
    for r in results:
        if r.learned_params:
            sn = short_names.get(r.name, r.name)
            for k, v in r.learned_params.items():
                if k in TRUE_PARAMS:
                    param_data.append((sn, k, TRUE_PARAMS[k], v))
    if param_data:
        labels = [f"{d[0]}\n{d[1]}" for d in param_data]
        true_vals = [d[2] for d in param_data]
        learned_vals = [d[3] for d in param_data]
        x = np.arange(len(labels))
        ax.bar(x - 0.15, true_vals, 0.3, label="True", color="#4CAF50", alpha=0.8)
        ax.bar(x + 0.15, learned_vals, 0.3, label="Learned", color="#E91E63", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("Parameter value")
        ax.set_title("Parameter Recovery")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Damped Pendulum: Compiled Gravity + Learned Dynamics",
                 fontsize=16, fontweight="bold", y=0.995)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results: list[PendulumResult], true_params: dict):
    print("\n" + "=" * 100)
    print("DAMPED PENDULUM ODE EXPERIMENT — RESULTS SUMMARY")
    print("=" * 100)

    header = f"{'Model':<28s} {'Params':>7s} {'MSE(in)':>12s} {'MSE(2x)':>12s} {'MSE(5x)':>12s}"
    print(header)
    print("-" * 75)
    for r in results:
        def fmt(v):
            return f"{v:.6f}" if math.isfinite(v) else "inf"
        print(f"{r.name:<28s} {r.n_params:>7d} {fmt(r.traj_mse_in):>12s} "
              f"{fmt(r.traj_mse_2x):>12s} {fmt(r.traj_mse_5x):>12s}")

    print("\nPARAMETER RECOVERY:")
    for r in results:
        if r.learned_params:
            print(f"\n  {r.name}:")
            for k, v in r.learned_params.items():
                if k in true_params:
                    err = r.param_errors.get(k, float("inf"))
                    print(f"    {k}: true={true_params[k]:.4f}, learned={v:.4f}, "
                          f"error={err:.4%}")

    print("\nIMPROVEMENT RATIOS (hybrid vs baselines):")
    hybrids = [r for r in results if "Hybrid" in r.name or "KnownStructure" in r.name]
    baselines = [r for r in results if "MLP" in r.name or "NeuralODE" in r.name]
    if hybrids and baselines:
        best_h = min(hybrids, key=lambda r: r.traj_mse_in)
        for bl in baselines:
            if best_h.traj_mse_in > 0 and math.isfinite(best_h.traj_mse_in):
                r_in = bl.traj_mse_in / best_h.traj_mse_in
                r_5x = bl.traj_mse_5x / max(best_h.traj_mse_5x, 1e-12) if math.isfinite(bl.traj_mse_5x) else float("inf")
                print(f"  Best hybrid vs {bl.name}: {r_in:.1f}x (in), {r_5x:.1f}x (5x)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Damped pendulum ODE experiment")
    parser.add_argument("--t-end", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--noise", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-size", type=int, default=25)
    parser.add_argument("--save-fig", default="examples/ode_damped_pendulum.png")
    parser.add_argument("--skip-neural-ode", action="store_true")
    parser.add_argument("--no-forcing", action="store_true",
                        help="Disable external forcing for Scenario 1")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip training; load saved data and regenerate figures")
    args = parser.parse_args()

    # --plot-only: load previously saved data and regenerate figures
    if args.plot_only:
        data_path = args.save_fig.replace(".png", "_data.pt")
        print(f"Loading saved data from {data_path}")
        data = torch.load(data_path, weights_only=False)
        true_params = data["true_params"]
        print_summary(data["results"], true_params)
        visualize(data["true_traj_in"], data["true_traj_5x"],
                  data["t_in"], data["t_5x"], data["results"], args.save_fig)
        return

    torch.manual_seed(args.seed)

    true_params = dict(TRUE_PARAMS)
    if args.no_forcing:
        true_params["F_amp"] = 0.0

    print("Damped Pendulum ODE Experiment")
    print(f"  True params: {true_params}")
    print(f"  Training horizon: [0, {args.t_end}]")
    print(f"  dt={args.dt}, epochs={args.epochs}, lr={args.lr}")
    print(f"  Noise: {args.noise*100:.0f}%, window_size={args.window_size}")
    print(f"  Forcing: {'OFF' if args.no_forcing else 'ON'}")
    print()

    # Generate ground truth
    t_in, traj_true = generate_trajectory(true_params, X0_DEFAULT, args.t_end, args.dt)
    t_5x, traj_true_5x = generate_trajectory(true_params, X0_DEFAULT, args.t_end * 5, args.dt)
    traj_noisy = add_noise(traj_true, args.noise, seed=args.seed)
    print(f"  Trajectory: {traj_true.shape[0]} steps")
    print(f"  theta range: [{traj_true[:,0].min():.3f}, {traj_true[:,0].max():.3f}]")
    print()

    results = []

    # Scenario 1: Known structure (no forcing)
    if args.no_forcing:
        print("=" * 60)
        print("SCENARIO 1: Known structure, learn g/L and b")
        print("=" * 60)
        m_rhs = compile_full_pendulum_rhs()
        model_s1 = PendulumKnownStructure(m_rhs)
        n_p = sum(p.numel() for p in model_s1.parameters() if p.requires_grad)
        print(f"  Trainable parameters: {n_p}")
        print(f"  Initial: g_over_L={-model_s1.neg_g_over_L.item():.2f}, b={-model_s1.neg_b.item():.2f}")
        print()

        torch.manual_seed(args.seed)
        t0 = time.time()
        losses, test_losses, test_epochs = train_model(
            model_s1, traj_noisy, X0_DEFAULT, args.t_end, args.dt,
            args.epochs, args.lr, traj_clean=traj_true, window_size=args.window_size)
        print(f"  Training time: {time.time()-t0:.1f}s")
        r = evaluate_model(model_s1, X0_DEFAULT, args.dt, args.t_end, true_params)
        r.losses = losses
        r.test_losses = test_losses
        r.test_epochs = test_epochs
        results.append(r)

    # Scenario 2: Compiled gravity + learned damping/forcing
    print("\n" + "=" * 60)
    print("SCENARIO 2: Compiled gravity + learned damping/forcing")
    print("=" * 60)
    m_gravity = compile_gravity_term()
    model_s2 = PendulumHybrid(m_gravity)
    n_p = sum(p.numel() for p in model_s2.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_p}")
    print()

    torch.manual_seed(args.seed)
    t0 = time.time()
    losses, test_losses, test_epochs = train_model(
        model_s2, traj_noisy, X0_DEFAULT, args.t_end, args.dt,
        args.epochs, args.lr * 0.3, traj_clean=traj_true, window_size=args.window_size)
    print(f"  Training time: {time.time()-t0:.1f}s")
    r = evaluate_model(model_s2, X0_DEFAULT, args.dt, args.t_end, true_params)
    r.name = "PendulumHybrid"
    r.losses = losses
    r.test_losses = test_losses
    r.test_epochs = test_epochs
    results.append(r)

    # Baseline: Pure MLP
    print("\n" + "=" * 60)
    print("BASELINE: Pure MLP RHS")
    print("=" * 60)
    model_mlp = PendulumMLP(hidden=64, layers=3)
    n_p = sum(p.numel() for p in model_mlp.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_p}")
    print()

    torch.manual_seed(args.seed)
    t0 = time.time()
    losses, test_losses, test_epochs = train_model(
        model_mlp, traj_noisy, X0_DEFAULT, args.t_end, args.dt,
        args.epochs, args.lr, traj_clean=traj_true, window_size=args.window_size)
    print(f"  Training time: {time.time()-t0:.1f}s")
    r = evaluate_model(model_mlp, X0_DEFAULT, args.dt, args.t_end, true_params)
    r.name = "PendulumMLP"
    r.losses = losses
    r.test_losses = test_losses
    r.test_epochs = test_epochs
    results.append(r)

    # Baseline: Neural ODE
    if not args.skip_neural_ode:
        print("\n" + "=" * 60)
        print("BASELINE: Neural ODE (torchdiffeq)")
        print("=" * 60)
        try:
            model_node = PendulumNeuralODE(hidden=64, layers=3)
            n_p = sum(p.numel() for p in model_node.parameters() if p.requires_grad)
            print(f"  Trainable parameters: {n_p}")
            print()

            torch.manual_seed(args.seed)
            t0 = time.time()
            losses, test_losses, test_epochs = train_neural_ode(
                model_node, traj_noisy, X0_DEFAULT, args.t_end,
                args.dt, args.epochs, args.lr,
                traj_clean=traj_true, window_size=args.window_size)
            print(f"  Training time: {time.time()-t0:.1f}s")
            r = evaluate_model(model_node, X0_DEFAULT, args.dt, args.t_end, true_params)
            r.name = "PendulumNeuralODE"
            r.losses = losses
            r.test_losses = test_losses
            r.test_epochs = test_epochs
            results.append(r)
        except Exception as e:
            print(f"  Neural ODE failed: {e}")

    # Baseline: Hand-coded PyTorch (Scenario 1 only — no forcing)
    if args.no_forcing:
        print("\n" + "=" * 60)
        print("BASELINE: Hand-coded PyTorch pendulum equations")
        print("=" * 60)
        model_hc = PendulumHandCoded()
        n_p = sum(p.numel() for p in model_hc.parameters() if p.requires_grad)
        print(f"  Trainable parameters: {n_p}")
        print(f"  Initial params: {model_hc.learned_params()}")
        print()

        torch.manual_seed(args.seed)
        t0 = time.time()
        losses, test_losses, test_epochs = train_model(
            model_hc, traj_noisy, X0_DEFAULT, args.t_end, args.dt,
            args.epochs, args.lr, traj_clean=traj_true, window_size=args.window_size)
        print(f"  Training time: {time.time()-t0:.1f}s")
        r = evaluate_model(model_hc, X0_DEFAULT, args.dt, args.t_end, true_params)
        r.name = "PendulumHandCoded"
        r.losses = losses
        r.test_losses = test_losses
        r.test_epochs = test_epochs
        results.append(r)

    # Baseline: PINN (Scenario 1 only — no forcing)
    if args.no_forcing:
        print("\n" + "=" * 60)
        print("BASELINE: Physics-Informed Neural Network (PINN)")
        print("=" * 60)
        model_pinn = PendulumPINN(hidden=64, layers=3)
        n_p = sum(p.numel() for p in model_pinn.parameters() if p.requires_grad)
        print(f"  Trainable parameters: {n_p}")
        print()

        torch.manual_seed(args.seed)
        t0 = time.time()
        losses, test_losses, test_epochs = train_pinn(
            model_pinn, traj_noisy, t_in, args.t_end,
            args.epochs, args.lr, traj_clean=traj_true, x0=X0_DEFAULT, dt=args.dt)
        print(f"  Training time: {time.time()-t0:.1f}s")
        r = evaluate_model(model_pinn, X0_DEFAULT, args.dt, args.t_end, true_params)
        r.name = "PendulumPINN"
        r.losses = losses
        r.test_losses = test_losses
        r.test_epochs = test_epochs
        results.append(r)

    # Save all data needed for plot regeneration
    data_path = args.save_fig.replace(".png", "_data.pt")
    torch.save({
        "true_traj_in": traj_true,
        "true_traj_5x": traj_true_5x,
        "t_in": t_in,
        "t_5x": t_5x,
        "results": results,
        "true_params": true_params,
    }, data_path)
    print(f"\nPlot data saved to {data_path}")

    print_summary(results, true_params)
    visualize(traj_true, traj_true_5x, t_in, t_5x, results, args.save_fig)


if __name__ == "__main__":
    main()

############################################################
#
# Neural Compiler: Compiling scheme into composable and
#                  differentiable neural network representations
#
# feynman_coefficient_learning.py: Learning physical constants from 15 Feynman equations with compiled structure
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Feynman Equation Coefficient Learning: compiled structure + trainable constants.

Demonstrates that when equation structure is KNOWN, compiling it as a frozen module
subgraph and learning only the physical constants achieves near-zero error with
perfect extrapolation — using 1-3 trainable parameters vs thousands for an MLP.

For each equation:
  1. Compile the equation structure as a frozen compiled module
  2. Physical constants become trainable nn.Parameters fed as inputs
  3. Train on noisy data to recover the true constants
  4. Compare against a pure MLP baseline

This is NOT symbolic regression (discovering equations). It demonstrates that
when domain knowledge IS available, compilation is dramatically more efficient
than approximation.
"""

import argparse
import math
import pickle
import sys
import time
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_compiler.compiler import compile_scheme
from neural_compiler.evaluator.direct_module import DirectModule


# ---------------------------------------------------------------------------
# Equation specifications
# ---------------------------------------------------------------------------

@dataclass
class EquationSpec:
    name: str
    description: str
    scheme: str
    input_vars: dict[str, tuple[float, float]]  # var_name -> (lo, hi)
    constants: dict[str, float]                  # const_name -> true_value
    category: str = ""


EQUATIONS: list[EquationSpec] = [
    # --- Simple arithmetic ---
    EquationSpec(
        name="Planck E=hf",
        description="E = h * f",
        scheme="(* h f)",
        input_vars={"f": (0.1, 10.0)},
        constants={"h": 6.626},
        category="Quantum",
    ),
    EquationSpec(
        name="Hooke F=-kx",
        description="F = -k * x",
        scheme="(* neg_k x)",
        input_vars={"x": (-3.0, 3.0)},
        constants={"neg_k": -2.5},
        category="Mechanics",
    ),
    EquationSpec(
        name="Kinetic energy",
        description="KE = alpha * m * v^2",
        scheme="(* alpha (* m (pow v 2)))",
        input_vars={"m": (0.5, 5.0), "v": (-3.0, 3.0)},
        constants={"alpha": 0.5},
        category="Mechanics",
    ),
    EquationSpec(
        name="Ideal gas P=nkT/V",
        description="P = n * kB * T / V",
        scheme="(/ (* n (* kB T)) V)",
        input_vars={"n": (0.5, 5.0), "T": (1.0, 10.0), "V": (0.5, 5.0)},
        constants={"kB": 1.381},
        category="Thermodynamics",
    ),
    # --- Power law ---
    EquationSpec(
        name="Gravity F=Gmm/r^2",
        description="F = G * m1 * m2 / r^2",
        scheme="(/ (* G (* m1 m2)) (pow r 2))",
        input_vars={"m1": (0.5, 5.0), "m2": (0.5, 5.0), "r": (0.5, 5.0)},
        constants={"G": 6.674},
        category="Mechanics",
    ),
    EquationSpec(
        name="Coulomb F=kqq/r^2",
        description="F = ke * q1 * q2 / r^2",
        scheme="(/ (* ke (* q1 q2)) (pow r 2))",
        input_vars={"q1": (0.5, 5.0), "q2": (0.5, 5.0), "r": (0.5, 5.0)},
        constants={"ke": 8.988},
        category="Electromagnetism",
    ),
    EquationSpec(
        name="E-field energy",
        description="u = coeff * E^2",
        scheme="(* coeff (pow Ef 2))",
        input_vars={"Ef": (0.1, 5.0)},
        constants={"coeff": 4.427},
        category="Electromagnetism",
    ),
    EquationSpec(
        name="Heat conduction",
        description="Q = kappa * (T2 - T1) * A / d",
        scheme="(/ (* kappa (* (- T2 T1) A)) d)",
        input_vars={"T2": (5.0, 15.0), "T1": (0.0, 5.0), "A": (0.5, 3.0), "d": (0.1, 2.0)},
        constants={"kappa": 1.5},
        category="Thermodynamics",
    ),
    # --- Square root ---
    EquationSpec(
        name="Speed of sound",
        description="v_s = sqrt(gamma * p / rho)",
        scheme="(sqrt (/ (* gamma pr) rho))",
        input_vars={"pr": (1.0, 10.0), "rho": (0.5, 5.0)},
        constants={"gamma": 1.4},
        category="Thermodynamics",
    ),
    EquationSpec(
        name="Pendulum period",
        description="T = two_pi * sqrt(L / g)",
        scheme="(* two_pi (sqrt (/ L g)))",
        input_vars={"L": (0.5, 5.0), "g": (5.0, 15.0)},
        constants={"two_pi": 6.283},
        category="Mechanics",
    ),
    # --- Relativistic (sqrt + pow) ---
    EquationSpec(
        name="Lorentz factor",
        description="gamma = m0 / sqrt(1 - v^2/c^2)",
        scheme="(/ m0 (sqrt (- 1 (pow (/ v c) 2))))",
        input_vars={"m0": (0.5, 5.0), "v": (0.1, 0.9)},
        constants={"c": 3.0},
        category="Relativity",
    ),
    EquationSpec(
        name="Relativistic energy",
        description="E = m * c^2 / sqrt(1 - v^2/c^2)",
        scheme="(/ (* m (pow c 2)) (sqrt (- 1 (pow (/ v c) 2))))",
        input_vars={"m": (0.5, 5.0), "v": (0.1, 0.9)},
        constants={"c": 3.0},
        category="Relativity",
    ),
    # --- Transcendental (sin, cos, exp) ---
    EquationSpec(
        name="Harmonic oscillator",
        description="x = A * sin(omega * t + phi)",
        scheme="(* A (sin (+ (* omega t) phi)))",
        input_vars={"t": (0.0, 6.0)},
        constants={"A": 2.0, "omega": 3.0, "phi": 0.5},
        category="Mechanics",
    ),
    EquationSpec(
        name="Gaussian",
        description="f = exp(-theta^2 / (2 * sigma^2))",
        scheme="(exp (/ (- 0 (pow theta 2)) (* 2 (pow sigma 2))))",
        input_vars={"theta": (-5.0, 5.0)},
        constants={"sigma": 2.0},
        category="Statistics",
    ),
    EquationSpec(
        name="Barometric formula",
        description="n = n0 * exp(-m*g*x / (kB*T))",
        scheme="(* n0 (exp (/ (* (- 0 m) (* g x)) (* kB T))))",
        input_vars={"m": (0.5, 2.0), "g": (5.0, 15.0), "x": (0.0, 3.0), "T": (1.0, 5.0)},
        constants={"n0": 5.0, "kB": 1.381},
        category="Thermodynamics",
    ),
]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class FeynmanHybrid(nn.Module):
    """Compiled equation structure + trainable physical constants."""

    def __init__(self, subgraph: DirectModule, constant_names: list[str],
                 init_values: list[float]):
        super().__init__()
        self.subgraph = subgraph
        self.constant_names = constant_names
        self.constants = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(val, dtype=torch.float32))
            for name, val in zip(constant_names, init_values)
        })

    def forward(self, data_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = next(iter(data_inputs.values())).shape[0]
        inputs = dict(data_inputs)
        for name, param in self.constants.items():
            inputs[name] = param.expand(batch_size)
        return self.subgraph.forward_batch(inputs)

    def learned_constants(self) -> dict[str, float]:
        return {name: p.item() for name, p in self.constants.items()}


# ---------------------------------------------------------------------------
# Hand-coded PyTorch baselines (no compilation)
# ---------------------------------------------------------------------------

EQUATION_FNS = {
    "Planck E=hf": lambda i: i["h"] * i["f"],
    "Hooke F=-kx": lambda i: i["neg_k"] * i["x"],
    "Kinetic energy": lambda i: i["alpha"] * i["m"] * i["v"] ** 2,
    "Ideal gas P=nkT/V": lambda i: i["n"] * i["kB"] * i["T"] / i["V"],
    "Gravity F=Gmm/r^2": lambda i: i["G"] * i["m1"] * i["m2"] / i["r"] ** 2,
    "Coulomb F=kqq/r^2": lambda i: i["ke"] * i["q1"] * i["q2"] / i["r"] ** 2,
    "E-field energy": lambda i: i["coeff"] * i["Ef"] ** 2,
    "Heat conduction": lambda i: i["kappa"] * (i["T2"] - i["T1"]) * i["A"] / i["d"],
    "Speed of sound": lambda i: torch.sqrt(i["gamma"] * i["pr"] / i["rho"]),
    "Pendulum period": lambda i: i["two_pi"] * torch.sqrt(i["L"] / i["g"]),
    "Lorentz factor": lambda i: i["m0"] / torch.sqrt(1 - (i["v"] / i["c"]) ** 2),
    "Relativistic energy": lambda i: i["m"] * i["c"] ** 2 / torch.sqrt(1 - (i["v"] / i["c"]) ** 2),
    "Harmonic oscillator": lambda i: i["A"] * torch.sin(i["omega"] * i["t"] + i["phi"]),
    "Gaussian": lambda i: torch.exp(-(i["theta"] ** 2) / (2 * i["sigma"] ** 2)),
    "Barometric formula": lambda i: i["n0"] * torch.exp(-i["m"] * i["g"] * i["x"] / (i["kB"] * i["T"])),
}


class HandCodedFeynman(nn.Module):
    """Hand-coded PyTorch equation — same structure as compiled, no compilation."""

    def __init__(self, equation_fn, constant_names: list[str],
                 init_values: list[float]):
        super().__init__()
        self.equation_fn = equation_fn
        self.constant_names = constant_names
        self.constants = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(val, dtype=torch.float32))
            for name, val in zip(constant_names, init_values)
        })

    def forward(self, data_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = next(iter(data_inputs.values())).shape[0]
        inputs = dict(data_inputs)
        for name, param in self.constants.items():
            inputs[name] = param.expand(batch_size)
        return self.equation_fn(inputs)

    def learned_constants(self) -> dict[str, float]:
        return {name: p.item() for name, p in self.constants.items()}


class FeynmanMLP(nn.Module):
    """Pure MLP baseline — must learn everything from scratch."""

    def __init__(self, n_inputs: int, hidden: int = 64, layers: int = 4):
        super().__init__()
        modules = [nn.Linear(n_inputs, hidden), nn.ReLU()]
        for _ in range(layers - 1):
            modules.extend([nn.Linear(hidden, hidden), nn.ReLU()])
        modules.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*modules)

    def forward(self, data_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        keys = sorted(data_inputs.keys())
        x = torch.stack([data_inputs[k] for k in keys], dim=1)
        return self.net(x).squeeze(1)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_data(eq: EquationSpec, n: int, noise_std: float = 0.0,
                  range_mult: float = 1.0, seed: int | None = None) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    if seed is not None:
        torch.manual_seed(seed)

    all_input_names = set(eq.input_vars.keys()) | set(eq.constants.keys())
    input_decl = {name: None for name in all_input_names}
    graph = compile_scheme(eq.scheme, inputs=input_decl)
    model = DirectModule(graph)

    data_inputs: dict[str, torch.Tensor] = {}
    for var, (lo, hi) in eq.input_vars.items():
        center = (lo + hi) / 2
        half_range = (hi - lo) / 2 * range_mult
        data_inputs[var] = torch.empty(n).uniform_(center - half_range, center + half_range)

    full_inputs = dict(data_inputs)
    for const_name, true_val in eq.constants.items():
        full_inputs[const_name] = torch.full((n,), true_val)

    with torch.no_grad():
        y_true = model.forward_batch(full_inputs)

    if noise_std > 0:
        y_true = y_true + noise_std * y_true.abs().mean() * torch.randn(n)

    return data_inputs, y_true


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def compile_equation(eq: EquationSpec) -> DirectModule:
    all_input_names = set(eq.input_vars.keys()) | set(eq.constants.keys())
    input_decl = {name: None for name in all_input_names}
    graph = compile_scheme(eq.scheme, inputs=input_decl)
    model = DirectModule(graph)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


def train_model(model: nn.Module, eq: EquationSpec, epochs: int, batch_size: int,
                lr: float, noise_std: float = 0.0) -> tuple[list[float], list[int], list[float]]:
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )

    # Generate held-out test data (no noise, separate seed)
    test_inputs, test_y = generate_data(eq, 2048, noise_std=0.0, seed=77777)

    eval_interval = max(1, epochs // 100)
    losses = []
    test_epochs: list[int] = []
    test_losses: list[float] = []

    for epoch in range(epochs):
        data_inputs, y_true = generate_data(eq, batch_size, noise_std=noise_std)
        y_pred = model(data_inputs)
        loss = F.mse_loss(y_pred, y_true)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if epoch % eval_interval == 0 or epoch == epochs - 1:
            with torch.no_grad():
                test_pred = model(test_inputs)
                test_loss = F.mse_loss(test_pred, test_y).item()
            test_epochs.append(epoch)
            test_losses.append(test_loss)

    return losses, test_epochs, test_losses


def evaluate_model(model: nn.Module, eq: EquationSpec, n: int = 10000,
                   range_mult: float = 1.0, seed: int = 999) -> float:
    data_inputs, y_true = generate_data(eq, n, range_mult=range_mult, seed=seed)
    with torch.no_grad():
        y_pred = model(data_inputs)
    return F.mse_loss(y_pred, y_true).item()


# ---------------------------------------------------------------------------
# Run one equation
# ---------------------------------------------------------------------------

@dataclass
class EquationResult:
    name: str
    category: str
    n_consts: int
    hybrid_params: int
    mlp_params: int
    handcoded_params: int
    true_constants: dict[str, float]
    learned_constants: dict[str, float]
    handcoded_learned_constants: dict[str, float]
    coeff_errors: dict[str, float]
    hybrid_mse_in: float
    mlp_mse_in: float
    handcoded_mse_in: float
    hybrid_mse_2x: float
    mlp_mse_2x: float
    handcoded_mse_2x: float
    hybrid_mse_5x: float
    mlp_mse_5x: float
    handcoded_mse_5x: float
    hybrid_losses: list[float] = field(default_factory=list)
    mlp_losses: list[float] = field(default_factory=list)
    handcoded_losses: list[float] = field(default_factory=list)
    hybrid_test_epochs: list[int] = field(default_factory=list)
    hybrid_test_losses: list[float] = field(default_factory=list)
    mlp_test_epochs: list[int] = field(default_factory=list)
    mlp_test_losses: list[float] = field(default_factory=list)
    handcoded_test_epochs: list[int] = field(default_factory=list)
    handcoded_test_losses: list[float] = field(default_factory=list)


def run_equation(eq: EquationSpec, epochs: int, batch_size: int, lr: float,
                 noise_std: float, seed: int) -> EquationResult:
    torch.manual_seed(seed)
    subgraph = compile_equation(eq)

    init_values = []
    for true_val in eq.constants.values():
        init_values.append(1.0)
    hybrid = FeynmanHybrid(subgraph, list(eq.constants.keys()), init_values)

    n_data_vars = len(eq.input_vars)
    mlp = FeynmanMLP(n_data_vars, hidden=64, layers=4)

    # Hand-coded baseline (same equation in pure PyTorch, no compilation)
    equation_fn = EQUATION_FNS[eq.name]
    handcoded = HandCodedFeynman(equation_fn, list(eq.constants.keys()), init_values)

    hybrid_params = sum(p.numel() for p in hybrid.parameters() if p.requires_grad)
    mlp_params = sum(p.numel() for p in mlp.parameters() if p.requires_grad)
    handcoded_params = sum(p.numel() for p in handcoded.parameters() if p.requires_grad)

    h_losses, h_test_epochs, h_test_losses = train_model(hybrid, eq, epochs, batch_size, lr, noise_std)
    torch.manual_seed(seed)
    m_losses, m_test_epochs, m_test_losses = train_model(mlp, eq, epochs, batch_size, lr, noise_std)
    torch.manual_seed(seed)
    hc_losses, hc_test_epochs, hc_test_losses = train_model(handcoded, eq, epochs, batch_size, lr, noise_std)

    h_mse_in = evaluate_model(hybrid, eq, range_mult=1.0)
    m_mse_in = evaluate_model(mlp, eq, range_mult=1.0)
    hc_mse_in = evaluate_model(handcoded, eq, range_mult=1.0)
    h_mse_2x = evaluate_model(hybrid, eq, range_mult=2.0)
    m_mse_2x = evaluate_model(mlp, eq, range_mult=2.0)
    hc_mse_2x = evaluate_model(handcoded, eq, range_mult=2.0)
    h_mse_5x = evaluate_model(hybrid, eq, range_mult=5.0)
    m_mse_5x = evaluate_model(mlp, eq, range_mult=5.0)
    hc_mse_5x = evaluate_model(handcoded, eq, range_mult=5.0)

    learned = hybrid.learned_constants()
    hc_learned = handcoded.learned_constants()
    coeff_errors = {}
    for name, true_val in eq.constants.items():
        learned_val = learned[name]
        if abs(true_val) > 1e-8:
            coeff_errors[name] = abs(learned_val - true_val) / abs(true_val)
        else:
            coeff_errors[name] = abs(learned_val - true_val)

    return EquationResult(
        name=eq.name,
        category=eq.category,
        n_consts=len(eq.constants),
        hybrid_params=hybrid_params,
        mlp_params=mlp_params,
        handcoded_params=handcoded_params,
        true_constants=dict(eq.constants),
        learned_constants=learned,
        handcoded_learned_constants=hc_learned,
        coeff_errors=coeff_errors,
        hybrid_mse_in=h_mse_in,
        mlp_mse_in=m_mse_in,
        handcoded_mse_in=hc_mse_in,
        hybrid_mse_2x=h_mse_2x,
        mlp_mse_2x=m_mse_2x,
        handcoded_mse_2x=hc_mse_2x,
        hybrid_mse_5x=h_mse_5x,
        mlp_mse_5x=m_mse_5x,
        handcoded_mse_5x=hc_mse_5x,
        hybrid_losses=h_losses,
        mlp_losses=m_losses,
        handcoded_losses=hc_losses,
        hybrid_test_epochs=h_test_epochs,
        hybrid_test_losses=h_test_losses,
        mlp_test_epochs=m_test_epochs,
        mlp_test_losses=m_test_losses,
        handcoded_test_epochs=hc_test_epochs,
        handcoded_test_losses=hc_test_losses,
    )


# ---------------------------------------------------------------------------
# Sample efficiency
# ---------------------------------------------------------------------------

def sample_efficiency(eq: EquationSpec, sample_sizes: list[int], epochs: int,
                      lr: float, seed: int) -> dict[str, list[float]]:
    """Train hybrid, MLP, and hand-coded at different training set sizes."""
    hybrid_mses = []
    mlp_mses = []
    handcoded_mses = []

    equation_fn = EQUATION_FNS[eq.name]

    for n_samples in sample_sizes:
        torch.manual_seed(seed)
        data_inputs, y_true = generate_data(eq, n_samples, seed=seed)

        subgraph = compile_equation(eq)
        init_values = [1.0] * len(eq.constants)
        hybrid = FeynmanHybrid(subgraph, list(eq.constants.keys()), init_values)
        n_data_vars = len(eq.input_vars)
        mlp = FeynmanMLP(n_data_vars, hidden=64, layers=4)
        handcoded = HandCodedFeynman(equation_fn, list(eq.constants.keys()), init_values)

        h_opt = torch.optim.Adam(hybrid.parameters(), lr=lr)
        m_opt = torch.optim.Adam(mlp.parameters(), lr=lr)
        hc_opt = torch.optim.Adam(handcoded.parameters(), lr=lr)

        for _ in range(epochs):
            y_h = hybrid(data_inputs)
            loss_h = F.mse_loss(y_h, y_true)
            h_opt.zero_grad()
            loss_h.backward()
            h_opt.step()

            y_m = mlp(data_inputs)
            loss_m = F.mse_loss(y_m, y_true)
            m_opt.zero_grad()
            loss_m.backward()
            m_opt.step()

            y_hc = handcoded(data_inputs)
            loss_hc = F.mse_loss(y_hc, y_true)
            hc_opt.zero_grad()
            loss_hc.backward()
            hc_opt.step()

        hybrid_mses.append(evaluate_model(hybrid, eq, range_mult=1.0))
        mlp_mses.append(evaluate_model(mlp, eq, range_mult=1.0))
        handcoded_mses.append(evaluate_model(handcoded, eq, range_mult=1.0))

    return {"hybrid": hybrid_mses, "mlp": mlp_mses, "handcoded": handcoded_mses}


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize(results: list[EquationResult], sample_eff: dict | None,
              save_path: str):
    n_eq = len(results)
    fig = plt.figure(figsize=(20, 24))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)

    # --- Panel 1: In-distribution MSE comparison (bar chart) ---
    ax = fig.add_subplot(gs[0, 0])
    names = [r.name for r in results]
    h_mse = [r.hybrid_mse_in for r in results]
    m_mse = [r.mlp_mse_in for r in results]
    hc_mse = [r.handcoded_mse_in for r in results]
    x = np.arange(n_eq)
    bar_h = 0.27
    ax.barh(x - bar_h, [max(v, 1e-12) for v in h_mse], bar_h, label="Compiled hybrid", color="#2196F3")
    ax.barh(x, [max(v, 1e-12) for v in hc_mse], bar_h, label="Hand-coded", color="#4CAF50")
    ax.barh(x + bar_h, [max(v, 1e-12) for v in m_mse], bar_h, label="MLP baseline", color="#FF9800")
    ax.set_xscale("log")
    ax.set_yticks(x)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("MSE (log scale)")
    ax.set_title("In-Distribution MSE")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()

    # --- Panel 2: Extrapolation MSE (5x range) ---
    ax = fig.add_subplot(gs[0, 1])
    h_mse5 = [r.hybrid_mse_5x for r in results]
    m_mse5 = [r.mlp_mse_5x for r in results]
    hc_mse5 = [r.handcoded_mse_5x for r in results]
    ax.barh(x - bar_h, [max(v, 1e-12) for v in h_mse5], bar_h, label="Compiled hybrid", color="#2196F3")
    ax.barh(x, [max(v, 1e-12) for v in hc_mse5], bar_h, label="Hand-coded", color="#4CAF50")
    ax.barh(x + bar_h, [max(v, 1e-12) for v in m_mse5], bar_h, label="MLP baseline", color="#FF9800")
    ax.set_xscale("log")
    ax.set_yticks(x)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("MSE (log scale)")
    ax.set_title("Extrapolation MSE (5x training range)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()

    # --- Panel 3: Coefficient recovery ---
    ax = fig.add_subplot(gs[1, 0])
    all_consts = []
    all_true = []
    all_learned = []
    all_labels = []
    for r in results:
        for name in r.true_constants:
            all_consts.append(f"{r.name}\n{name}")
            all_true.append(r.true_constants[name])
            all_learned.append(r.learned_constants[name])
            all_labels.append(name)
    cx = np.arange(len(all_consts))
    ax.bar(cx - 0.15, all_true, 0.3, label="True", color="#4CAF50", alpha=0.8)
    ax.bar(cx + 0.15, all_learned, 0.3, label="Learned", color="#E91E63", alpha=0.8)
    ax.set_xticks(cx)
    ax.set_xticklabels(all_consts, fontsize=6, rotation=45, ha="right")
    ax.set_ylabel("Constant value")
    ax.set_title("Coefficient Recovery: True vs Learned")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # --- Panel 4: Relative coefficient error ---
    ax = fig.add_subplot(gs[1, 1])
    errors = []
    error_labels = []
    for r in results:
        for name, err in r.coeff_errors.items():
            errors.append(max(err, 1e-10))
            error_labels.append(f"{r.name}\n{name}")
    cx = np.arange(len(errors))
    colors = ["#4CAF50" if e < 0.01 else "#FF9800" if e < 0.1 else "#F44336" for e in errors]
    ax.bar(cx, errors, color=colors)
    ax.set_yscale("log")
    ax.set_xticks(cx)
    ax.set_xticklabels(error_labels, fontsize=6, rotation=45, ha="right")
    ax.set_ylabel("Relative error |learned - true| / |true|")
    ax.set_title("Coefficient Recovery Error")
    ax.axhline(0.01, color="green", linestyle="--", alpha=0.5, label="1% error")
    ax.axhline(0.1, color="orange", linestyle="--", alpha=0.5, label="10% error")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # --- Panel 5: Parameter count comparison ---
    ax = fig.add_subplot(gs[2, 0])
    h_params = [r.hybrid_params for r in results]
    m_params = [r.mlp_params for r in results]
    ax.barh(x - 0.2, h_params, 0.4, label="Compiled hybrid", color="#2196F3")
    ax.barh(x + 0.2, m_params, 0.4, label="MLP baseline", color="#FF9800")
    ax.set_xscale("log")
    ax.set_yticks(x)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Trainable parameters (log scale)")
    ax.set_title("Parameter Count")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()

    # --- Panel 6: Improvement ratios ---
    ax = fig.add_subplot(gs[2, 1])
    in_ratios = [r.mlp_mse_in / max(r.hybrid_mse_in, 1e-12) for r in results]
    ex_ratios = [r.mlp_mse_5x / max(r.hybrid_mse_5x, 1e-12) for r in results]
    ax.barh(x - 0.2, in_ratios, 0.4, label="In-distribution", color="#2196F3")
    ax.barh(x + 0.2, ex_ratios, 0.4, label="Extrapolation (5x)", color="#FF9800")
    ax.set_xscale("log")
    ax.set_yticks(x)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Improvement ratio (MLP MSE / Hybrid MSE)")
    ax.set_title("Hybrid Improvement over MLP")
    ax.axvline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()

    # --- Panel 7: Sample efficiency (if available) ---
    if sample_eff is not None:
        ax = fig.add_subplot(gs[3, 0])
        sizes = sample_eff["sizes"]
        ax.loglog(sizes, sample_eff["hybrid"], "o-", color="#2196F3", linewidth=2, label="Compiled hybrid")
        ax.loglog(sizes, sample_eff["handcoded"], "^-", color="#4CAF50", linewidth=2, label="Hand-coded")
        ax.loglog(sizes, sample_eff["mlp"], "s-", color="#FF9800", linewidth=2, label="MLP baseline")
        ax.set_xlabel("Training set size")
        ax.set_ylabel("Test MSE")
        ax.set_title(f"Sample Efficiency ({sample_eff['eq_name']})")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Feynman Equation Coefficient Learning: Compiled Structure vs MLP",
                 fontsize=16, fontweight="bold", y=0.995)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(results: list[EquationResult]):
    print("\n" + "=" * 160)
    print("FEYNMAN EQUATION COEFFICIENT LEARNING — RESULTS SUMMARY")
    print("=" * 160)

    header = (f"{'Equation':<25s} {'Cat':<14s} {'#C':>3s} {'H-par':>5s} {'HC-par':>6s} {'MLP-par':>7s} "
              f"{'H-MSE(in)':>10s} {'HC-MSE(in)':>11s} {'M-MSE(in)':>10s} {'M/H Ratio':>9s} "
              f"{'H-MSE(5x)':>10s} {'HC-MSE(5x)':>11s} {'M-MSE(5x)':>10s} {'M/H Ratio':>9s}")
    print(header)
    print("-" * 160)

    total_hybrid_wins_in = 0
    total_hybrid_wins_ex = 0
    median_in_ratios = []
    median_ex_ratios = []

    for r in results:
        in_ratio = r.mlp_mse_in / max(r.hybrid_mse_in, 1e-12)
        ex_ratio = r.mlp_mse_5x / max(r.hybrid_mse_5x, 1e-12)
        median_in_ratios.append(in_ratio)
        median_ex_ratios.append(ex_ratio)
        if in_ratio > 1:
            total_hybrid_wins_in += 1
        if ex_ratio > 1:
            total_hybrid_wins_ex += 1

        print(f"{r.name:<25s} {r.category:<14s} {r.n_consts:>3d} {r.hybrid_params:>5d} {r.handcoded_params:>6d} {r.mlp_params:>7d} "
              f"{r.hybrid_mse_in:>10.2e} {r.handcoded_mse_in:>11.2e} {r.mlp_mse_in:>10.2e} {in_ratio:>9.1f}x "
              f"{r.hybrid_mse_5x:>10.2e} {r.handcoded_mse_5x:>11.2e} {r.mlp_mse_5x:>10.2e} {ex_ratio:>9.1f}x")

    print("-" * 160)
    print(f"\nHybrid wins vs MLP (in-dist): {total_hybrid_wins_in}/{len(results)}")
    print(f"Hybrid wins vs MLP (extrap):  {total_hybrid_wins_ex}/{len(results)}")
    valid_in = [r for r in median_in_ratios if np.isfinite(r)]
    valid_ex = [r for r in median_ex_ratios if np.isfinite(r)]
    print(f"Median improvement vs MLP (in-dist): {np.median(valid_in):.1f}x")
    print(f"Median improvement vs MLP (extrap):  {np.median(valid_ex):.1f}x")

    # Compiled vs hand-coded comparison
    hc_diffs_in = [abs(r.hybrid_mse_in - r.handcoded_mse_in) for r in results]
    hc_diffs_5x = [abs(r.hybrid_mse_5x - r.handcoded_mse_5x) for r in results]
    print(f"\nCompiled vs Hand-coded (same equations, same params):")
    print(f"  Mean |MSE difference| (in-dist): {np.mean(hc_diffs_in):.2e}")
    print(f"  Mean |MSE difference| (extrap):  {np.mean(hc_diffs_5x):.2e}")

    print("\nCOEFFICIENT RECOVERY:")
    print(f"{'Equation':<25s} {'Constant':<10s} {'True':>10s} {'Learned':>10s} {'RelError':>10s} {'Status'}")
    print("-" * 80)
    for r in results:
        for name in r.true_constants:
            err = r.coeff_errors[name]
            status = "EXACT" if err < 0.001 else "GOOD" if err < 0.01 else "OK" if err < 0.1 else "POOR"
            print(f"{r.name:<25s} {name:<10s} {r.true_constants[name]:>10.4f} "
                  f"{r.learned_constants[name]:>10.4f} {err:>10.6f} {status}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Feynman equation coefficient learning")
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--noise", type=float, default=0.01, help="Relative noise std")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-efficiency", action="store_true",
                        help="Run sample efficiency analysis")
    parser.add_argument("--save-fig", default="examples/feynman_coefficient_learning.png")
    parser.add_argument("--equations", type=str, default=None,
                        help="Comma-separated indices (0-based) to run subset")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip training; load saved data and regenerate figures")
    args = parser.parse_args()

    if args.plot_only:
        data_path = args.save_fig.replace(".png", "_data.pkl")
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        results = data["results"]
        sample_eff = data.get("sample_eff")
        print_summary(results)
        visualize(results, sample_eff, args.save_fig)
        return

    if args.equations is not None:
        indices = [int(i) for i in args.equations.split(",")]
        equations = [EQUATIONS[i] for i in indices]
    else:
        equations = EQUATIONS

    print(f"Feynman Equation Coefficient Learning")
    print(f"  Equations: {len(equations)}")
    print(f"  Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    print(f"  Noise: {args.noise * 100:.0f}%, Seed: {args.seed}")
    print()

    results: list[EquationResult] = []
    for i, eq in enumerate(equations):
        print(f"[{i+1}/{len(equations)}] {eq.name}: {eq.description}")
        print(f"  Scheme: {eq.scheme}")
        print(f"  Constants: {eq.constants}")
        t0 = time.time()
        r = run_equation(eq, args.epochs, args.batch_size, args.lr, args.noise, args.seed)
        dt = time.time() - t0
        in_ratio = r.mlp_mse_in / max(r.hybrid_mse_in, 1e-12)
        print(f"  Hybrid:     {r.hybrid_params} params, MSE={r.hybrid_mse_in:.2e} (in), {r.hybrid_mse_5x:.2e} (5x)")
        print(f"  Hand-coded: {r.handcoded_params} params, MSE={r.handcoded_mse_in:.2e} (in), {r.handcoded_mse_5x:.2e} (5x)")
        print(f"  MLP:        {r.mlp_params} params, MSE={r.mlp_mse_in:.2e} (in), {r.mlp_mse_5x:.2e} (5x)")
        print(f"  Improvement vs MLP: {in_ratio:.1f}x (in-dist), {r.mlp_mse_5x / max(r.hybrid_mse_5x, 1e-12):.1f}x (5x)")
        print(f"  Learned (compiled):   {r.learned_constants}")
        print(f"  Learned (hand-coded): {r.handcoded_learned_constants}")
        print(f"  Time: {dt:.1f}s")
        print()
        results.append(r)

    print_summary(results)

    sample_eff = None
    if args.sample_efficiency:
        eq_idx = len(equations) // 2
        eq = equations[eq_idx]
        print(f"\nSample efficiency analysis on: {eq.name}")
        sizes = [10, 50, 100, 500, 1000, 5000, 10000]
        eff = sample_efficiency(eq, sizes, epochs=args.epochs, lr=args.lr, seed=args.seed)
        sample_eff = {"sizes": sizes, "eq_name": eq.name, **eff}
        print(f"  Sizes:      {sizes}")
        print(f"  Hybrid:     {['%.2e' % v for v in eff['hybrid']]}")
        print(f"  Hand-coded: {['%.2e' % v for v in eff['handcoded']]}")
        print(f"  MLP:        {['%.2e' % v for v in eff['mlp']]}")

    data_path = args.save_fig.replace(".png", "_data.pkl")
    with open(data_path, "wb") as f:
        pickle.dump({"results": results, "sample_eff": sample_eff}, f)
    print(f"Results saved to {data_path}")

    visualize(results, sample_eff, args.save_fig)


if __name__ == "__main__":
    main()

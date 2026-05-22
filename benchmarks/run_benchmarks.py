#!/usr/bin/env python3
############################################################
#
# Neural Compiler: Compiling scheme into composable and
#                  differentiable neural network representations
#
# run_benchmarks.py: Performance benchmark suite across evaluators, batch sizes, and program complexities
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Benchmark suite for the neural compiler.

Measures performance across evaluators, batch sizes, program complexities,
and compares against Python native and Guile Scheme baselines.

Usage:
    python -m benchmarks.run_benchmarks                # full suite, CSV to stdout
    python -m benchmarks.run_benchmarks --quick         # fewer repeats/sizes
    python -m benchmarks.run_benchmarks --batch-only    # only batch scaling
    python -m benchmarks.run_benchmarks --eval-only     # only evaluator comparison
    python -m benchmarks.run_benchmarks --scale-only    # only complexity scaling
    python -m benchmarks.run_benchmarks --baseline-only # only Python/Scheme baselines
    python -m benchmarks.run_benchmarks -o results.csv  # write CSV to file
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass

import numpy as np
import torch

from neural_compiler.compiler import compile_scheme
from neural_compiler.evaluator import evaluate, SchemeGNN, DirectModule


# ---------------------------------------------------------------------------
# Program suite
# ---------------------------------------------------------------------------

@dataclass
class Program:
    name: str
    source: str
    inputs: list[str]
    description: str


PROGRAMS = [
    Program("add", "(+ x y)", ["x", "y"], "2-input addition"),
    Program("square_plus", "(+ (* x x) y)", ["x", "y"], "x^2 + y"),
    Program("four_ops", "(+ (* a b) (- c d))", ["a", "b", "c", "d"], "a*b + c-d"),
    Program("abs", "(if (> x 0) x (- 0 x))", ["x"], "absolute value"),
    Program(
        "clamp",
        "(if (< x 0) 0 (if (> x 10) 10 x))",
        ["x"],
        "clamp to [0,10]",
    ),
    Program(
        "quadratic",
        "(+ (+ (* a (* x x)) (* b x)) c)",
        ["x", "a", "b", "c"],
        "ax^2 + bx + c",
    ),
    Program(
        "discriminant",
        "(let ((d (- (* b b) (* 4 (* a c))))) (if (>= d 0) 1 0))",
        ["a", "b", "c"],
        "sign of b^2-4ac",
    ),
    Program(
        "dist_sq",
        "(let ((dx (- x1 x2)) (dy (- y1 y2))) (+ (* dx dx) (* dy dy)))",
        ["x1", "y1", "x2", "y2"],
        "squared distance",
    ),
    Program(
        "dot4",
        "(+ (+ (* a1 b1) (* a2 b2)) (+ (* a3 b3) (* a4 b4)))",
        ["a1", "a2", "a3", "a4", "b1", "b2", "b3", "b4"],
        "4-elem dot product",
    ),
]


# ---------------------------------------------------------------------------
# Python native baselines — hand-written equivalents
# ---------------------------------------------------------------------------

_PY_SCALAR: dict[str, callable] = {
    "add":           lambda d: d["x"] + d["y"],
    "square_plus":   lambda d: d["x"] * d["x"] + d["y"],
    "four_ops":      lambda d: d["a"] * d["b"] + (d["c"] - d["d"]),
    "abs":           lambda d: d["x"] if d["x"] > 0 else -d["x"],
    "clamp":         lambda d: 0.0 if d["x"] < 0 else (10.0 if d["x"] > 10 else d["x"]),
    "quadratic":     lambda d: d["a"] * d["x"] * d["x"] + d["b"] * d["x"] + d["c"],
    "discriminant":  lambda d: 1.0 if d["b"] * d["b"] - 4 * d["a"] * d["c"] >= 0 else 0.0,
    "dist_sq":       lambda d: (d["x1"] - d["x2"]) ** 2 + (d["y1"] - d["y2"]) ** 2,
    "dot4":          lambda d: d["a1"]*d["b1"] + d["a2"]*d["b2"] + d["a3"]*d["b3"] + d["a4"]*d["b4"],
}

_TORCH_BATCH: dict[str, callable] = {
    "add":           lambda d: d["x"] + d["y"],
    "square_plus":   lambda d: d["x"] * d["x"] + d["y"],
    "four_ops":      lambda d: d["a"] * d["b"] + (d["c"] - d["d"]),
    "abs":           lambda d: torch.where(d["x"] > 0, d["x"], -d["x"]),
    "clamp":         lambda d: torch.clamp(d["x"], 0.0, 10.0),
    "quadratic":     lambda d: d["a"] * d["x"] * d["x"] + d["b"] * d["x"] + d["c"],
    "discriminant":  lambda d: torch.where(
                         d["b"] * d["b"] - 4 * d["a"] * d["c"] >= 0,
                         torch.ones_like(d["a"]), torch.zeros_like(d["a"])),
    "dist_sq":       lambda d: (d["x1"] - d["x2"]) ** 2 + (d["y1"] - d["y2"]) ** 2,
    "dot4":          lambda d: d["a1"]*d["b1"] + d["a2"]*d["b2"] + d["a3"]*d["b3"] + d["a4"]*d["b4"],
}

_NP_BATCH: dict[str, callable] = {
    "add":           lambda d: d["x"] + d["y"],
    "square_plus":   lambda d: d["x"] * d["x"] + d["y"],
    "four_ops":      lambda d: d["a"] * d["b"] + (d["c"] - d["d"]),
    "abs":           lambda d: np.where(d["x"] > 0, d["x"], -d["x"]),
    "clamp":         lambda d: np.clip(d["x"], 0.0, 10.0),
    "quadratic":     lambda d: d["a"] * d["x"] * d["x"] + d["b"] * d["x"] + d["c"],
    "discriminant":  lambda d: np.where(
                         d["b"] * d["b"] - 4 * d["a"] * d["c"] >= 0, 1.0, 0.0),
    "dist_sq":       lambda d: (d["x1"] - d["x2"]) ** 2 + (d["y1"] - d["y2"]) ** 2,
    "dot4":          lambda d: d["a1"]*d["b1"] + d["a2"]*d["b2"] + d["a3"]*d["b3"] + d["a4"]*d["b4"],
}


def make_chain(depth: int) -> Program:
    expr = "x"
    for _ in range(depth):
        expr = f"(+ {expr} 1)"
    return Program(f"chain_{depth}", expr, ["x"], f"chain of {depth} adds")


def make_tree(width: int) -> Program:
    vars_needed = 2 * width
    var_names = [f"v{i}" for i in range(vars_needed)]
    products = [f"(* {var_names[2*i]} {var_names[2*i+1]})" for i in range(width)]
    while len(products) > 1:
        new = []
        for i in range(0, len(products) - 1, 2):
            new.append(f"(+ {products[i]} {products[i+1]})")
        if len(products) % 2 == 1:
            new.append(products[-1])
        products = new
    return Program(f"tree_{width}", products[0], var_names, f"tree of {width} muls")


# ---------------------------------------------------------------------------
# Timing utilities
# ---------------------------------------------------------------------------

def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def bench(fn, warmup: int = 5, repeats: int = 30, device: torch.device | None = None) -> dict:
    for _ in range(warmup):
        fn()
    if device:
        _sync(device)

    times = []
    for _ in range(repeats):
        if device:
            _sync(device)
        t0 = time.perf_counter()
        fn()
        if device:
            _sync(device)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return {
        "mean_s": statistics.mean(times),
        "std_s": statistics.stdev(times) if len(times) > 1 else 0.0,
        "min_s": min(times),
        "median_s": statistics.median(times),
    }


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _find_guile() -> str | None:
    return shutil.which("guile")


def _make_row(benchmark, program, evaluator, device, batch_size, nodes, depth, t, throughput="") -> dict:
    return {
        "benchmark": benchmark,
        "program": program,
        "evaluator": evaluator,
        "device": str(device),
        "batch_size": batch_size,
        "nodes": nodes,
        "depth": depth,
        "mean_s": f"{t['mean_s']:.6e}",
        "std_s": f"{t['std_s']:.6e}",
        "min_s": f"{t['min_s']:.6e}",
        "median_s": f"{t['median_s']:.6e}",
        "throughput": f"{throughput:.1f}" if isinstance(throughput, (int, float)) else throughput,
    }


# ---------------------------------------------------------------------------
# Benchmark 1: Evaluator comparison (single input)
# ---------------------------------------------------------------------------

def bench_evaluators(programs: list[Program], repeats: int, writer: csv.DictWriter) -> list[dict]:
    log = _log("EVALUATOR COMPARISON")
    gpu = _get_device()
    results = []

    for prog in programs:
        input_decl = {n: None for n in prog.inputs}
        graph = compile_scheme(prog.source, inputs=input_decl)
        vals = {n: 2.0 for n in prog.inputs}
        torch_vals = {n: torch.tensor(2.0) for n in prog.inputs}
        gpu_vals = {n: torch.tensor(2.0, device=gpu) for n in prog.inputs}

        gnn = SchemeGNN(graph)
        pyg_cpu = DirectModule(graph)
        pyg_gpu = DirectModule(graph).to(gpu) if gpu.type != "cpu" else None

        evaluators = [
            ("sequential", lambda: evaluate(graph, vals), None),
            ("SchemeGNN", lambda: gnn(torch_vals), None),
            ("Direct_CPU", lambda: pyg_cpu(torch_vals), None),
        ]
        if pyg_gpu is not None:
            evaluators.append(("Direct_GPU", lambda: pyg_gpu(gpu_vals), gpu))

        for eval_name, fn, dev in evaluators:
            t = bench(fn, warmup=5, repeats=repeats, device=dev)
            row = _make_row("evaluator", prog.name, eval_name, dev or "cpu",
                            1, len(graph.nodes), graph.depth(), t, 1.0 / t["mean_s"])
            writer.writerow(row)
            results.append(row)

        log(f"  {prog.name:20s}  seq={results[-len(evaluators)]['mean_s']}  "
            f"GNN={results[-len(evaluators)+1]['mean_s']}  "
            f"Direct={results[-len(evaluators)+2]['mean_s']}"
            + (f"  GPU={results[-1]['mean_s']}" if pyg_gpu else ""))

    return results


# ---------------------------------------------------------------------------
# Benchmark 2: Batch throughput scaling
# ---------------------------------------------------------------------------

def bench_batch_scaling(
    programs: list[Program],
    batch_sizes: list[int],
    repeats: int,
    writer: csv.DictWriter,
) -> list[dict]:
    log = _log("BATCH THROUGHPUT SCALING")
    gpu = _get_device()
    results = []

    for prog in programs:
        input_decl = {n: None for n in prog.inputs}
        graph = compile_scheme(prog.source, inputs=input_decl)

        pyg_cpu = DirectModule(graph)
        pyg_gpu = DirectModule(graph).to(gpu) if gpu.type != "cpu" else None

        for bs in batch_sizes:
            for dev_name, model, device in [("cpu", pyg_cpu, torch.device("cpu"))] + (
                [("gpu", pyg_gpu, gpu)] if pyg_gpu else []
            ):
                inputs_b = {n: torch.randn(bs, device=device) for n in prog.inputs}
                try:
                    t = bench(
                        lambda: model.forward_batch(inputs_b),
                        warmup=3, repeats=repeats,
                        device=device if device.type != "cpu" else None,
                    )
                except RuntimeError:
                    continue

                throughput = bs / t["mean_s"]
                row = _make_row("batch_scaling", prog.name, f"batch_{dev_name}",
                                device, bs, len(graph.nodes), graph.depth(), t, throughput)
                writer.writerow(row)
                results.append(row)

            cpu_row = next(
                (r for r in results if r["program"] == prog.name
                 and r["batch_size"] == bs and r["evaluator"] == "batch_cpu"),
                None,
            )
            gpu_row = next(
                (r for r in results if r["program"] == prog.name
                 and r["batch_size"] == bs and r["evaluator"] == "batch_gpu"),
                None,
            )
            cpu_tp = cpu_row["throughput"] if cpu_row else "—"
            gpu_tp = gpu_row["throughput"] if gpu_row else "—"
            log(f"  {prog.name:20s}  B={bs:<8d}  CPU={cpu_tp:>12s}/s  GPU={gpu_tp:>12s}/s")

    return results


# ---------------------------------------------------------------------------
# Benchmark 3: Program complexity scaling
# ---------------------------------------------------------------------------

def bench_complexity_scaling(
    depths: list[int],
    widths: list[int],
    repeats: int,
    writer: csv.DictWriter,
) -> list[dict]:
    log = _log("COMPLEXITY SCALING")
    gpu = _get_device()
    results = []
    batch_size = 1000

    log("  --- Depth scaling (chain of adds, batch=1000) ---")
    for d in depths:
        prog = make_chain(d)
        input_decl = {n: None for n in prog.inputs}
        graph = compile_scheme(prog.source, inputs=input_decl)
        n_nodes = len(graph.nodes)
        g_depth = graph.depth()

        pyg = DirectModule(graph)
        inputs_b = {n: torch.randn(batch_size) for n in prog.inputs}

        t = bench(lambda: pyg.forward_batch(inputs_b), warmup=3, repeats=repeats)
        throughput = batch_size / t["mean_s"]
        row = _make_row("depth_scaling", prog.name, "batch_cpu", "cpu",
                        batch_size, n_nodes, g_depth, t, throughput)
        writer.writerow(row)
        results.append(row)

        if gpu.type != "cpu":
            pyg_gpu = DirectModule(graph).to(gpu)
            inputs_gpu = {n: torch.randn(batch_size, device=gpu) for n in prog.inputs}
            t_gpu = bench(
                lambda: pyg_gpu.forward_batch(inputs_gpu),
                warmup=3, repeats=repeats, device=gpu,
            )
            tp_gpu = batch_size / t_gpu["mean_s"]
            row_gpu = _make_row("depth_scaling", prog.name, "batch_gpu", str(gpu),
                                batch_size, n_nodes, g_depth, t_gpu, tp_gpu)
            writer.writerow(row_gpu)
            results.append(row_gpu)
            log(f"  depth={d:<4d}  nodes={n_nodes:<5d}  CPU={throughput:>10.0f}/s  GPU={tp_gpu:>10.0f}/s")
        else:
            log(f"  depth={d:<4d}  nodes={n_nodes:<5d}  CPU={throughput:>10.0f}/s")

    log("  --- Width scaling (balanced tree of muls, batch=1000) ---")
    for w in widths:
        prog = make_tree(w)
        input_decl = {n: None for n in prog.inputs}
        graph = compile_scheme(prog.source, inputs=input_decl)
        n_nodes = len(graph.nodes)
        g_depth = graph.depth()

        pyg = DirectModule(graph)
        inputs_b = {n: torch.randn(batch_size) for n in prog.inputs}

        t = bench(lambda: pyg.forward_batch(inputs_b), warmup=3, repeats=repeats)
        throughput = batch_size / t["mean_s"]
        row = _make_row("width_scaling", prog.name, "batch_cpu", "cpu",
                        batch_size, n_nodes, g_depth, t, throughput)
        writer.writerow(row)
        results.append(row)

        if gpu.type != "cpu":
            pyg_gpu = DirectModule(graph).to(gpu)
            inputs_gpu = {n: torch.randn(batch_size, device=gpu) for n in prog.inputs}
            t_gpu = bench(
                lambda: pyg_gpu.forward_batch(inputs_gpu),
                warmup=3, repeats=repeats, device=gpu,
            )
            tp_gpu = batch_size / t_gpu["mean_s"]
            row_gpu = _make_row("width_scaling", prog.name, "batch_gpu", str(gpu),
                                batch_size, n_nodes, g_depth, t_gpu, tp_gpu)
            writer.writerow(row_gpu)
            results.append(row_gpu)
            log(f"  width={w:<4d}  nodes={n_nodes:<5d}  depth={g_depth:<3d}  CPU={throughput:>10.0f}/s  GPU={tp_gpu:>10.0f}/s")
        else:
            log(f"  width={w:<4d}  nodes={n_nodes:<5d}  depth={g_depth:<3d}  CPU={throughput:>10.0f}/s")

    return results


# ---------------------------------------------------------------------------
# Benchmark 4: Compilation time
# ---------------------------------------------------------------------------

def bench_compilation(programs: list[Program], repeats: int, writer: csv.DictWriter) -> list[dict]:
    log = _log("COMPILATION TIME")
    results = []

    for prog in programs:
        input_decl = {n: None for n in prog.inputs}

        t_compile = bench(
            lambda: compile_scheme(prog.source, inputs=input_decl),
            warmup=3, repeats=repeats,
        )

        graph = compile_scheme(prog.source, inputs=input_decl)
        t_model = bench(
            lambda: DirectModule(graph),
            warmup=3, repeats=repeats,
        )

        for phase, t in [("compile", t_compile), ("model_init", t_model)]:
            row = _make_row("compilation", prog.name, phase, "cpu",
                            0, len(graph.nodes), graph.depth(), t)
            writer.writerow(row)
            results.append(row)

        log(f"  {prog.name:20s}  compile={t_compile['mean_s']:.4e}s  model_init={t_model['mean_s']:.4e}s")

    return results


# ---------------------------------------------------------------------------
# Benchmark 5: Python native baselines (scalar + batched)
# ---------------------------------------------------------------------------

def bench_python_baseline(
    programs: list[Program],
    batch_sizes: list[int],
    repeats: int,
    writer: csv.DictWriter,
) -> list[dict]:
    log = _log("PYTHON / NUMPY / TORCH BASELINES")
    gpu = _get_device()
    results = []

    for prog in programs:
        if prog.name not in _PY_SCALAR:
            continue

        vals = {n: 2.0 for n in prog.inputs}
        py_fn = _PY_SCALAR[prog.name]
        graph = compile_scheme(prog.source, inputs={n: None for n in prog.inputs})
        n_nodes = len(graph.nodes)
        g_depth = graph.depth()

        # --- Python scalar ---
        t = bench(lambda: py_fn(vals), warmup=10, repeats=repeats)
        row = _make_row("baseline_single", prog.name, "python", "cpu",
                        1, n_nodes, g_depth, t, 1.0 / t["mean_s"])
        writer.writerow(row)
        results.append(row)
        py_tp = f"{1.0 / t['mean_s']:.0f}"

        log(f"  {prog.name:20s}  python_scalar={t['mean_s']:.3e}s  ({py_tp}/s)")

        # --- Batched: numpy, torch CPU, torch GPU ---
        np_fn = _NP_BATCH[prog.name]
        pt_fn = _TORCH_BATCH[prog.name]

        for bs in batch_sizes:
            np_vals = {n: np.random.randn(bs).astype(np.float32) for n in prog.inputs}
            pt_vals = {n: torch.randn(bs) for n in prog.inputs}

            # NumPy
            t_np = bench(lambda: np_fn(np_vals), warmup=3, repeats=repeats)
            tp_np = bs / t_np["mean_s"]
            row_np = _make_row("baseline_batch", prog.name, "numpy", "cpu",
                               bs, n_nodes, g_depth, t_np, tp_np)
            writer.writerow(row_np)
            results.append(row_np)

            # PyTorch CPU
            t_pt = bench(lambda: pt_fn(pt_vals), warmup=3, repeats=repeats)
            tp_pt = bs / t_pt["mean_s"]
            row_pt = _make_row("baseline_batch", prog.name, "torch_cpu", "cpu",
                               bs, n_nodes, g_depth, t_pt, tp_pt)
            writer.writerow(row_pt)
            results.append(row_pt)

            # PyTorch GPU
            if gpu.type != "cpu":
                gpu_vals = {n: torch.randn(bs, device=gpu) for n in prog.inputs}
                t_gpu = bench(
                    lambda: pt_fn(gpu_vals),
                    warmup=3, repeats=repeats, device=gpu,
                )
                tp_gpu = bs / t_gpu["mean_s"]
                row_gpu = _make_row("baseline_batch", prog.name, "torch_gpu", str(gpu),
                                    bs, n_nodes, g_depth, t_gpu, tp_gpu)
                writer.writerow(row_gpu)
                results.append(row_gpu)

            log(f"    B={bs:<8d}  np={tp_np:>12.0f}/s  torch={tp_pt:>12.0f}/s"
                + (f"  torch_gpu={tp_gpu:>12.0f}/s" if gpu.type != "cpu" else ""))

    return results


# ---------------------------------------------------------------------------
# Benchmark 6: Guile Scheme interpreter
# ---------------------------------------------------------------------------

_GUILE_TEMPLATE = """\
(use-modules (ice-9 time))

(define result-acc 0.0)

(define (run-bench n)
  (let loop ((i n))
    (when (> i 0)
      (set! result-acc
        (let ({bindings})
          {expr}))
      (loop (- i 1)))))

;; Warmup
(run-bench {warmup_n})

;; Timed run
(let* ((n {n})
       (t0 (get-internal-real-time))
       (ignored (run-bench n))
       (t1 (get-internal-real-time))
       (elapsed (/ (- t1 t0) internal-time-units-per-second 1.0)))
  (display elapsed)
  (newline))
"""


def _run_guile(guile_path: str, source: str, inputs: list[str],
               n: int = 1000000, warmup_n: int = 100000,
               timeout: float = 30.0) -> float | None:
    bindings = " ".join(f"({name} 2.0)" for name in inputs)
    script = _GUILE_TEMPLATE.format(
        bindings=bindings, expr=source, n=n, warmup_n=warmup_n,
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".scm", delete=False) as f:
        f.write(script)
        f.flush()
        script_path = f.name

    try:
        result = subprocess.run(
            [guile_path, "--no-auto-compile", script_path],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            print(f"    Guile error: {result.stderr.strip()}", file=sys.stderr)
            return None
        elapsed = float(result.stdout.strip())
        return elapsed
    except (subprocess.TimeoutExpired, ValueError) as e:
        print(f"    Guile failed: {e}", file=sys.stderr)
        return None
    finally:
        os.unlink(script_path)


def bench_scheme(
    programs: list[Program],
    repeats: int,
    writer: csv.DictWriter,
) -> list[dict]:
    guile_path = _find_guile()
    if guile_path is None:
        print("  Guile not found, skipping Scheme baseline", file=sys.stderr)
        return []

    log = _log(f"GUILE SCHEME BASELINE ({guile_path})")
    results = []

    n_iters = 1000000

    for prog in programs:
        graph = compile_scheme(prog.source, inputs={n: None for n in prog.inputs})
        n_nodes = len(graph.nodes)
        g_depth = graph.depth()

        times = []
        for run_idx in range(repeats):
            elapsed = _run_guile(guile_path, prog.source, prog.inputs,
                                 n=n_iters, warmup_n=n_iters // 10)
            if elapsed is not None:
                times.append(elapsed)

        if not times:
            log(f"  {prog.name:20s}  FAILED")
            continue

        per_eval = statistics.mean(times) / n_iters
        t = {
            "mean_s": per_eval,
            "std_s": statistics.stdev([t / n_iters for t in times]) if len(times) > 1 else 0.0,
            "min_s": min(times) / n_iters,
            "median_s": statistics.median(times) / n_iters,
        }
        throughput = 1.0 / per_eval
        row = _make_row("baseline_single", prog.name, "guile", "cpu",
                        1, n_nodes, g_depth, t, throughput)
        writer.writerow(row)
        results.append(row)

        log(f"  {prog.name:20s}  {per_eval:.3e}s/eval  ({throughput:,.0f}/s)")

    return results


# ---------------------------------------------------------------------------
# Large program helpers — programmatic baselines for generated programs
# ---------------------------------------------------------------------------

def _make_tree_scalar_fn(width: int):
    def fn(vals):
        products = [vals[f"v{2*i}"] * vals[f"v{2*i+1}"] for i in range(width)]
        while len(products) > 1:
            new = []
            for i in range(0, len(products) - 1, 2):
                new.append(products[i] + products[i+1])
            if len(products) % 2 == 1:
                new.append(products[-1])
            products = new
        return products[0]
    return fn


def _make_chain_scalar_fn(depth: int):
    def fn(vals):
        result = vals["x"]
        for _ in range(depth):
            result = result + 1.0
        return result
    return fn


def _make_tree_np_fn(width: int):
    def fn(vals):
        products = [vals[f"v{2*i}"] * vals[f"v{2*i+1}"] for i in range(width)]
        while len(products) > 1:
            new = []
            for i in range(0, len(products) - 1, 2):
                new.append(products[i] + products[i+1])
            if len(products) % 2 == 1:
                new.append(products[-1])
            products = new
        return products[0]
    return fn


def _make_chain_np_fn(depth: int):
    def fn(vals):
        result = vals["x"]
        for _ in range(depth):
            result = result + 1.0
        return result
    return fn


def _make_tree_torch_fn(width: int):
    def fn(vals):
        products = [vals[f"v{2*i}"] * vals[f"v{2*i+1}"] for i in range(width)]
        while len(products) > 1:
            new = []
            for i in range(0, len(products) - 1, 2):
                new.append(products[i] + products[i+1])
            if len(products) % 2 == 1:
                new.append(products[-1])
            products = new
        return products[0]
    return fn


def _make_chain_torch_fn(depth: int):
    def fn(vals):
        result = vals["x"]
        for _ in range(depth):
            result = result + 1.0
        return result
    return fn


# ---------------------------------------------------------------------------
# Benchmark 7: Large programs — trees & chains at scale
# ---------------------------------------------------------------------------

def bench_large_programs(
    tree_widths: list[int],
    chain_depths: list[int],
    repeats: int,
    writer: csv.DictWriter,
    batch_multiplier: int = 1,
) -> list[dict]:
    log = _log("LARGE PROGRAM BENCHMARKS")
    gpu = _get_device()
    results = []

    sys.setrecursionlimit(100000)

    if gpu.type == "cuda":
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
    elif gpu.type == "mps":
        vram_bytes = 8 * 1024**3
    else:
        vram_bytes = 4 * 1024**3

    vram_budget = int(vram_bytes * 0.75)

    def _pick_batch(n_nodes, n_inputs):
        mem_per_sample = (n_nodes + n_inputs) * 4 * 4
        max_batch = vram_budget // max(mem_per_sample, 1)
        max_batch = min(max_batch, 10_000_000)
        max_batch = max(max_batch, 1000)
        nice = 10 ** int(np.log10(max(max_batch, 1)))
        if max_batch >= 5 * nice:
            return 5 * nice
        return nice

    # --- Trees ---
    log("  --- Wide trees: width × (mul + sum reduction) ---")
    for w in tree_widths:
        prog = make_tree(w)
        input_decl = {n: None for n in prog.inputs}
        graph = compile_scheme(prog.source, inputs=input_decl)
        n_nodes = len(graph.nodes)
        n_inputs = len(prog.inputs)
        g_depth = graph.depth()
        batch_size = _pick_batch(n_nodes, n_inputs)

        log(f"  tree_{w}: {n_nodes} nodes, depth={g_depth}, batch={batch_size}")

        # Python scalar loop
        py_fn = _make_tree_scalar_fn(w)
        scalar_vals = {n: 2.0 for n in prog.inputs}
        n_scalar = max(100, min(10000, int(1.0 / max(1e-9, w * 5e-8))))
        py_fn(scalar_vals)  # warmup
        t0 = time.perf_counter()
        for _ in range(n_scalar):
            py_fn(scalar_vals)
        py_elapsed = time.perf_counter() - t0
        py_per_eval = py_elapsed / n_scalar
        py_tp = 1.0 / py_per_eval
        t_py = {"mean_s": py_per_eval, "std_s": 0.0, "min_s": py_per_eval, "median_s": py_per_eval}
        row = _make_row("large_program", prog.name, "python_scalar", "cpu",
                        1, n_nodes, g_depth, t_py, py_tp)
        writer.writerow(row)
        results.append(row)

        # NumPy vectorized batch
        np_fn = _make_tree_np_fn(w)
        np_vals = {n: np.random.randn(batch_size).astype(np.float32) for n in prog.inputs}
        t_np = bench(lambda: np_fn(np_vals), warmup=3, repeats=repeats)
        np_tp = batch_size / t_np["mean_s"]
        row = _make_row("large_program", prog.name, "numpy", "cpu",
                        batch_size, n_nodes, g_depth, t_np, np_tp)
        writer.writerow(row)
        results.append(row)

        # PyTorch CPU vectorized batch
        pt_fn = _make_tree_torch_fn(w)
        pt_vals = {n: torch.randn(batch_size) for n in prog.inputs}
        t_pt = bench(lambda: pt_fn(pt_vals), warmup=3, repeats=repeats)
        pt_tp = batch_size / t_pt["mean_s"]
        row = _make_row("large_program", prog.name, "torch_cpu", "cpu",
                        batch_size, n_nodes, g_depth, t_pt, pt_tp)
        writer.writerow(row)
        results.append(row)

        # GNN batch CPU
        model_cpu = DirectModule(graph)
        gnn_vals_cpu = {n: torch.randn(batch_size) for n in prog.inputs}
        t_gnn_cpu = bench(
            lambda: model_cpu.forward_batch(gnn_vals_cpu),
            warmup=3, repeats=min(repeats, 10),
        )
        gnn_cpu_tp = batch_size / t_gnn_cpu["mean_s"]
        row = _make_row("large_program", prog.name, "batch_cpu", "cpu",
                        batch_size, n_nodes, g_depth, t_gnn_cpu, gnn_cpu_tp)
        writer.writerow(row)
        results.append(row)

        # GNN batch GPU
        if gpu.type != "cpu":
            model_gpu = DirectModule(graph).to(gpu)
            gnn_vals_gpu = {n: torch.randn(batch_size, device=gpu) for n in prog.inputs}
            t_gnn_gpu = bench(
                lambda: model_gpu.forward_batch(gnn_vals_gpu),
                warmup=3, repeats=min(repeats, 10),
                device=gpu,
            )
            gnn_gpu_tp = batch_size / t_gnn_gpu["mean_s"]
            row = _make_row("large_program", prog.name, "batch_gpu", str(gpu),
                            batch_size, n_nodes, g_depth, t_gnn_gpu, gnn_gpu_tp)
            writer.writerow(row)
            results.append(row)

        # PyTorch GPU vectorized batch
        if gpu.type != "cpu":
            gpu_pt_vals = {n: torch.randn(batch_size, device=gpu) for n in prog.inputs}
            t_gpu_pt = bench(
                lambda: pt_fn(gpu_pt_vals),
                warmup=3, repeats=repeats, device=gpu,
            )
            gpu_pt_tp = batch_size / t_gpu_pt["mean_s"]
            row = _make_row("large_program", prog.name, "torch_gpu", str(gpu),
                            batch_size, n_nodes, g_depth, t_gpu_pt, gpu_pt_tp)
            writer.writerow(row)
            results.append(row)

        speedup_vs_py = gnn_cpu_tp / py_tp
        log(f"    python={py_tp:>8,.0f}/s  np={np_tp:>10,.0f}/s  "
            f"gnn_cpu={gnn_cpu_tp:>8,.0f}/s  "
            + (f"gnn_gpu={gnn_gpu_tp:>8,.0f}/s  " if gpu.type != "cpu" else "")
            + f"speedup_vs_py={speedup_vs_py:.1f}x")

        if gpu.type == "cuda":
            torch.cuda.empty_cache()

    # --- Chains ---
    log("  --- Deep chains: x + 1 + 1 + ... (depth MP rounds) ---")
    for d in chain_depths:
        prog = make_chain(d)
        input_decl = {n: None for n in prog.inputs}
        graph = compile_scheme(prog.source, inputs=input_decl)
        n_nodes = len(graph.nodes)
        n_inputs = len(prog.inputs)
        g_depth = graph.depth()
        batch_size = _pick_batch(n_nodes, n_inputs)

        log(f"  chain_{d}: {n_nodes} nodes, depth={g_depth}, batch={batch_size}")

        # Python scalar loop
        py_fn = _make_chain_scalar_fn(d)
        scalar_vals = {"x": 2.0}
        n_scalar = max(1000, min(100000, int(1.0 / max(1e-9, d * 2e-8))))
        py_fn(scalar_vals)
        t0 = time.perf_counter()
        for _ in range(n_scalar):
            py_fn(scalar_vals)
        py_elapsed = time.perf_counter() - t0
        py_per_eval = py_elapsed / n_scalar
        py_tp = 1.0 / py_per_eval
        t_py = {"mean_s": py_per_eval, "std_s": 0.0, "min_s": py_per_eval, "median_s": py_per_eval}
        row = _make_row("large_program", prog.name, "python_scalar", "cpu",
                        1, n_nodes, g_depth, t_py, py_tp)
        writer.writerow(row)
        results.append(row)

        # NumPy vectorized batch
        np_fn = _make_chain_np_fn(d)
        np_vals = {"x": np.random.randn(batch_size).astype(np.float32)}
        t_np = bench(lambda: np_fn(np_vals), warmup=3, repeats=repeats)
        np_tp = batch_size / t_np["mean_s"]
        row = _make_row("large_program", prog.name, "numpy", "cpu",
                        batch_size, n_nodes, g_depth, t_np, np_tp)
        writer.writerow(row)
        results.append(row)

        # PyTorch CPU vectorized batch
        pt_fn = _make_chain_torch_fn(d)
        pt_vals = {"x": torch.randn(batch_size)}
        t_pt = bench(lambda: pt_fn(pt_vals), warmup=3, repeats=repeats)
        pt_tp = batch_size / t_pt["mean_s"]
        row = _make_row("large_program", prog.name, "torch_cpu", "cpu",
                        batch_size, n_nodes, g_depth, t_pt, pt_tp)
        writer.writerow(row)
        results.append(row)

        # GNN batch CPU — skip if depth > 250 (too slow)
        if d <= 250:
            model_cpu = DirectModule(graph)
            gnn_vals_cpu = {"x": torch.randn(batch_size)}
            t_gnn_cpu = bench(
                lambda: model_cpu.forward_batch(gnn_vals_cpu),
                warmup=2, repeats=min(repeats, 5),
            )
            gnn_cpu_tp = batch_size / t_gnn_cpu["mean_s"]
            row = _make_row("large_program", prog.name, "batch_cpu", "cpu",
                            batch_size, n_nodes, g_depth, t_gnn_cpu, gnn_cpu_tp)
            writer.writerow(row)
            results.append(row)

            if gpu.type != "cpu":
                model_gpu = DirectModule(graph).to(gpu)
                gnn_vals_gpu = {"x": torch.randn(batch_size, device=gpu)}
                t_gnn_gpu = bench(
                    lambda: model_gpu.forward_batch(gnn_vals_gpu),
                    warmup=2, repeats=min(repeats, 5),
                    device=gpu,
                )
                gnn_gpu_tp = batch_size / t_gnn_gpu["mean_s"]
                row = _make_row("large_program", prog.name, "batch_gpu", str(gpu),
                                batch_size, n_nodes, g_depth, t_gnn_gpu, gnn_gpu_tp)
                writer.writerow(row)
                results.append(row)

            log(f"    python={py_tp:>8,.0f}/s  np={np_tp:>10,.0f}/s  "
                f"gnn_cpu={gnn_cpu_tp:>8,.0f}/s  "
                + (f"gnn_gpu={gnn_gpu_tp:>8,.0f}/s" if gpu.type != "cpu" else ""))
        else:
            log(f"    python={py_tp:>8,.0f}/s  np={np_tp:>10,.0f}/s  "
                f"gnn: skipped (depth={d} MP rounds too slow)")

        # PyTorch GPU vectorized batch
        if gpu.type != "cpu":
            gpu_pt_vals = {"x": torch.randn(batch_size, device=gpu)}
            t_gpu_pt = bench(
                lambda: pt_fn(gpu_pt_vals),
                warmup=3, repeats=repeats, device=gpu,
            )
            gpu_pt_tp = batch_size / t_gpu_pt["mean_s"]
            row = _make_row("large_program", prog.name, "torch_gpu", str(gpu),
                            batch_size, n_nodes, g_depth, t_gpu_pt, gpu_pt_tp)
            writer.writerow(row)
            results.append(row)

        if gpu.type == "cuda":
            torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

_PROGRESS_FILE = os.environ.get("BENCH_PROGRESS_FILE")

def _log(header: str | None = None):
    if header:
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"  {header}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        sys.stderr.flush()
    def _emit(msg):
        print(msg, file=sys.stderr)
        sys.stderr.flush()
        if _PROGRESS_FILE:
            with open(_PROGRESS_FILE, "a") as f:
                f.write(msg + "\n")
                f.flush()
                os.fsync(f.fileno())
    return _emit


CSV_FIELDS = [
    "benchmark", "program", "evaluator", "device", "batch_size",
    "nodes", "depth", "mean_s", "std_s", "min_s", "median_s", "throughput",
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Neural compiler benchmarks")
    parser.add_argument("--quick", action="store_true", help="Fewer repeats and sizes")
    parser.add_argument("--batch-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--scale-only", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--scheme-only", action="store_true")
    parser.add_argument("--large-only", action="store_true", help="Only large program benchmarks")
    parser.add_argument("--batch-multiplier", type=int, default=1,
                        help="Multiply default batch sizes (e.g. 100 for 24GB GPU)")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output CSV file")
    args = parser.parse_args()

    run_all = not any([
        args.batch_only, args.eval_only, args.scale_only,
        args.compile_only, args.baseline_only, args.scheme_only,
        args.large_only,
    ])

    bm = args.batch_multiplier
    if args.quick:
        repeats = 5
        batch_sizes = [1, 100, 10000 * bm]
        depths = [1, 4, 16, 64]
        widths = [1, 4, 16, 64]
        scheme_repeats = 3
        large_tree_widths = [100, 1000]
        large_chain_depths = [50, 250]
    else:
        repeats = 30
        batch_sizes = [1, 10, 100, 1000, 10000, 100000 * bm]
        if bm > 1:
            batch_sizes.extend([1000000 * bm, 10000000])
        depths = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        widths = [1, 2, 4, 8, 16, 32, 64, 128]
        scheme_repeats = 5
        large_tree_widths = [50, 100, 150, 200, 250, 300]
        large_chain_depths = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

    outfile = open(args.output, "w", newline="", buffering=1) if args.output else sys.stdout
    writer = csv.DictWriter(outfile, fieldnames=CSV_FIELDS)
    writer.writeheader()

    gpu = _get_device()
    print(f"\nDevice: {gpu}", file=sys.stderr)
    if gpu.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}", file=sys.stderr)
    print(f"Repeats: {repeats} (scheme: {scheme_repeats})", file=sys.stderr)

    if run_all or args.eval_only:
        bench_evaluators(PROGRAMS, repeats, writer)

    if run_all or args.batch_only:
        bench_batch_scaling(PROGRAMS, batch_sizes, repeats, writer)

    if run_all or args.scale_only:
        bench_complexity_scaling(depths, widths, repeats, writer)

    if run_all or args.compile_only:
        bench_compilation(PROGRAMS, repeats, writer)

    if run_all or args.baseline_only:
        bench_python_baseline(PROGRAMS, batch_sizes, repeats, writer)

    if run_all or args.baseline_only or args.scheme_only:
        bench_scheme(PROGRAMS, scheme_repeats, writer)

    if run_all or args.large_only:
        bench_large_programs(large_tree_widths, large_chain_depths, repeats, writer,
                             batch_multiplier=args.batch_multiplier)

    if args.output:
        outfile.close()
        print(f"\nResults written to {args.output}", file=sys.stderr)

    print(f"\nDone.", file=sys.stderr)


if __name__ == "__main__":
    main()

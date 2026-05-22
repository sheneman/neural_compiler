# The Neural Compiler

A compiler that translates programs written in a first-order expression language with Scheme syntax into frozen, differentiable PyTorch `nn.Module` objects. Compiled modules compute *exactly* -- they produce the same output as the source program to floating-point precision, with exact gradients via autograd, and can be embedded in hybrid architectures alongside trainable neural networks.

**Paper:** [The Neural Compiler: Compiling Symbolic Programs into Differentiable Modules for Hybrid Scientific Machine Learning](https://arxiv.org/abs/XXXX.XXXXX)

## Key Ideas

- **Exact computation.** Compiled modules produce numerically identical results to hand-coded PyTorch -- confirmed to machine precision across all experiments.
- **Frozen + trainable.** Known physics is compiled and frozen; unknown components (parameters, correction terms) are learned. The compiled module contributes zero approximation error; error arises only from learned components.
- **Composable.** Chains of compiled modules maintain zero error at arbitrary depth, while neural approximation errors amplify multiplicatively.
- **From text.** Changing `"(sin x)"` to `"(exp (sin x))"` produces a new correct module instantly. No hand-coding required.

## Architecture

```
Scheme source  -->  Parse  -->  AST  -->  ANF  -->  TCO  -->  ComputeGraph  -->  DirectModule
                                                                                 (nn.Module)
```

The compiler transforms source code through four stages: parsing to an abstract syntax tree, flattening to A-Normal Form (one operation per let-binding), tail-call optimization of recursive functions to iterative loops, and compilation to a `DirectModule` that evaluates nodes in topological order via instruction dispatch.

## Installation

```bash
pip install -e .
```

Requires Python 3.10+ and PyTorch 2.0+.

## Quick Start

### One-shot evaluation

```python
from neural_compiler.compiler import run_scheme

run_scheme("(+ (* 3 4) 5)")                                    # 17.0
run_scheme("(+ (* 3 x) (- y 1))", {"x": 4.0, "y": 7.0})      # 18.0
run_scheme("(if (> x 0) x (- 0 x))", {"x": -5.0})             # 5.0
```

### Compile to a reusable nn.Module

```python
import torch
from neural_compiler.compiler import compile_scheme
from neural_compiler.evaluator import DirectModule

graph = compile_scheme("(+ (* a b) (- c d))",
                       inputs={"a": None, "b": None, "c": None, "d": None})
model = DirectModule(graph)

# Single evaluation
result = model({"a": torch.tensor(3.0), "b": torch.tensor(4.0),
                "c": torch.tensor(10.0), "d": torch.tensor(3.0)})  # tensor(19.)

# Batched evaluation (GPU-compatible)
result = model.forward_batch({
    "a": torch.tensor([3.0, 1.0, 5.0]),
    "b": torch.tensor([4.0, 2.0, 12.0]),
    "c": torch.tensor([10.0, 5.0, 100.0]),
    "d": torch.tensor([3.0, 1.0, 40.0]),
})  # tensor([19., 6., 120.])
```

### Hybrid architecture: compiled physics + learned parameters

```python
import torch
import torch.nn as nn
from neural_compiler.compiler import compile_scheme
from neural_compiler.evaluator import DirectModule

# Compile known physics: F = -k * x (Hooke's law)
graph = compile_scheme("(* neg_k x)", inputs={"neg_k": None, "x": None})
compiled = DirectModule(graph)
compiled.freeze()  # freeze the compiled structure

# Trainable parameter
k = nn.Parameter(torch.tensor(-1.0))  # will learn the true spring constant

# Training loop learns k from data while the equation structure is exact
```

### Loops and recursion

```python
from neural_compiler.compiler import run_scheme

# Factorial via loop/recur
run_scheme("(loop ((n 10) (acc 1)) (if (= n 0) acc (recur (- n 1) (* acc n))))")
# 3628800.0

# Fibonacci via letrec
run_scheme("""
  (letrec ((fib (lambda (n) (if (< n 2) n (+ (fib (- n 1)) (fib (- n 2)))))))
    (fib 10))
""")  # 55.0
```

### Vector and matrix operations

```python
from neural_compiler.compiler import run_scheme

# Dot product
run_scheme("(dot [1 2 3] [4 5 6])")  # 32.0

# Matrix-vector multiply (identity matrix)
run_scheme("(matvec (mat [1 0] [0 1]) [3 4])")  # [3.0, 4.0]

# Cross product
run_scheme("(cross [1 0 0] [0 1 0])")  # [0.0, 0.0, 1.0]
```

## Supported Operations

The language supports 51 operations in four categories:

| Category | Operations |
|----------|-----------|
| **Arithmetic** (12) | `+`, `-`, `*`, `/`, `pow`, `abs`, `min`, `max`, `modulo`, `remainder`, `sqrt`, `log`, `exp`, `sin`, `cos` |
| **Comparison & Logic** (8) | `=`, `<`, `>`, `<=`, `>=`, `and`, `or`, `not` |
| **Vector** (9) | `vec`, `ref`, `dot`, `cross`, `norm`, `normalize`, `vsum`, `scale`, `vlen` |
| **Matrix** (11) | `mat`, `matmul`, `matvec`, `transpose`, `trace`, `det`, `inv`, `outer`, `eye`, `zeros`, `ones` |
| **Control flow** (7) | `let`, `if`, `lambda`, `loop`/`recur`, `letrec`/`call` |

Bracket syntax `[1 2 3]` desugars to `(vec 1 2 3)` for concise vector literals.

## Experiments

Six experiments from the paper are included in `examples/`:

| Experiment | Script | Domain |
|-----------|--------|--------|
| Feynman coefficient learning | `feynman_coefficient_learning.py` | 15 algebraic physics laws |
| Lotka-Volterra ODE | `ode_lotka_volterra.py` | Predator-prey dynamics |
| Damped pendulum ODE | `ode_damped_pendulum.py` | Transcendental dynamics (sin) |
| 1D heat equation PDE | `pde_heat_equation.py` | Finite-difference diffusion |
| 3D vector mechanics | `vector_mechanics.py` | Gravitational force (vector ops) |
| Compositional generalization | `compositional_generalization.py` | Error propagation in chains |

Additional examples demonstrate hybrid architectures: routing, deep composition, residual connections, CNN+physics, and more.

## Testing

```bash
pytest tests/ -v
```

589 tests covering parsing, ANF transformation, tail-call optimization, graph construction, primitive operations, sequential and DirectModule evaluation, vector/matrix operations, batched execution, loops, recursion, and composition patterns.

## Project Structure

```
neural_compiler/
  parser/          # Tokenizer, recursive-descent parser, AST nodes
  anf/             # A-Normal Form transformation, TCO pass
  graph/           # Dataflow graph construction from ANF
  ops/             # Primitive operation implementations (PyTorch)
  evaluator/       # DirectModule (primary), sequential engine, GNN module
  compiler.py      # Top-level compile_scheme() and run_scheme() API
examples/          # Paper experiments and hybrid architecture demos
tests/             # Unit and integration tests
benchmarks/        # Performance benchmarking suite
```

## License

MIT

## Citation

```bibtex
@article{sheneman2025neuralcompiler,
  title={The Neural Compiler: Compiling Symbolic Programs into Differentiable Modules for Hybrid Scientific Machine Learning},
  author={Sheneman, Luke},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

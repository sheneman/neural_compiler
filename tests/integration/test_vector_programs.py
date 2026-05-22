############################################################
#
# Neural Compiler: Compiling scheme into composable and
#                  differentiable neural network representations
#
# test_vector_programs.py: Integration tests for vector/matrix programs with analytic ground truth
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Integration tests: end-to-end compiled vector/matrix programs.

These tests exercise realistic physics-style programs through the full
neural compiler pipeline: Scheme source -> compile -> evaluate/DirectModule.
Vector and matrix operations are tested with known analytic results.
"""

import math
import pytest
import torch
import numpy as np

from neural_compiler.compiler import compile_scheme
from neural_compiler.evaluator import DirectModule, evaluate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _eval_both(source, inputs_val, inputs_decl=None):
    """Compile and evaluate with both the sequential engine and DirectModule.

    Returns (sequential_result, direct_result) as tensors.
    """
    if inputs_decl is None:
        inputs_decl = {k: None for k in inputs_val}
    graph = compile_scheme(source, inputs=inputs_decl)

    seq_result = evaluate(graph, inputs_val)
    if not isinstance(seq_result, torch.Tensor):
        seq_result = torch.tensor(seq_result, dtype=torch.float32)

    model = DirectModule(graph)
    tensor_inputs = {}
    for k, v in inputs_val.items():
        if isinstance(v, torch.Tensor):
            tensor_inputs[k] = v
        else:
            tensor_inputs[k] = torch.tensor(v, dtype=torch.float32)
    direct_result = model(tensor_inputs)
    if direct_result.dim() == 0:
        direct_result = direct_result.unsqueeze(0)
    if seq_result.dim() == 0:
        seq_result = seq_result.unsqueeze(0)

    return seq_result, direct_result


# ---------------------------------------------------------------------------
# 1. TestVectorPhysics
# ---------------------------------------------------------------------------

class TestVectorPhysics:
    """Physics-motivated vector programs: gravity, kinetic energy, springs."""

    def test_gravitational_force(self):
        """F = -G / |r|^3 * r for a known position vector.

        Scheme: (scale (/ (- 0 G) (pow (norm r) 3)) r)
        """
        source = "(scale (/ (- 0 G) (pow (norm r) 3)) r)"
        r = torch.tensor([3.0, 4.0, 0.0])
        G = torch.tensor(6.674)
        inputs_val = {"G": G, "r": r}

        seq, direct = _eval_both(source, inputs_val)

        # Manual: |r| = 5, |r|^3 = 125, -G/125 = -0.053392, F = scale * r
        norm_r = torch.norm(r)
        expected = (-G / norm_r**3) * r

        assert torch.allclose(seq, expected, atol=1e-4), f"seq {seq} != expected {expected}"
        assert torch.allclose(direct, expected, atol=1e-4), f"direct {direct} != expected {expected}"

    def test_kinetic_energy(self):
        """KE = 0.5 * dot(v, v) -- scalar result from vector input.

        Scheme: (* 0.5 (dot v v))
        """
        source = "(* 0.5 (dot v v))"
        v = torch.tensor([3.0, 4.0, 0.0])
        inputs_val = {"v": v}

        graph = compile_scheme(source, inputs={"v": None})
        seq_result = evaluate(graph, inputs_val)
        model = DirectModule(graph)
        direct_result = model({"v": v})

        expected = 0.5 * torch.dot(v, v)
        assert seq_result == pytest.approx(expected.item(), rel=1e-5)
        assert direct_result.item() == pytest.approx(expected.item(), rel=1e-5)

    def test_spring_force_3d(self):
        """Hooke's law: F = -k * r in 3D.

        Scheme: (scale (- 0 k) r)
        """
        source = "(scale (- 0 k) r)"
        k = torch.tensor(2.5)
        r = torch.tensor([1.0, -2.0, 3.0])
        inputs_val = {"k": k, "r": r}

        seq, direct = _eval_both(source, inputs_val)

        expected = -k * r
        assert torch.allclose(seq, expected, atol=1e-5)
        assert torch.allclose(direct, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# 2. TestMatrixPhysics
# ---------------------------------------------------------------------------

class TestMatrixPhysics:
    """Physics-motivated matrix programs: rotation, stress-strain."""

    def test_rotation_matvec(self):
        """Apply a 2D rotation matrix to a vector.

        R = [[cos_t, -sin_t], [sin_t, cos_t]]
        Scheme: (matvec (mat (vec cos_t (- 0 sin_t)) (vec sin_t cos_t)) v)

        Pass cos_t, sin_t, v as inputs.
        """
        source = "(matvec (mat (vec cos_t (- 0 sin_t)) (vec sin_t cos_t)) v)"
        theta = math.pi / 4  # 45 degrees
        cos_t = torch.tensor(math.cos(theta))
        sin_t = torch.tensor(math.sin(theta))
        v = torch.tensor([1.0, 0.0])
        inputs_val = {"cos_t": cos_t, "sin_t": sin_t, "v": v}

        seq, direct = _eval_both(source, inputs_val)

        # Expected: rotate [1, 0] by 45 deg -> [cos(45), sin(45)]
        expected = torch.tensor([math.cos(theta), math.sin(theta)])
        assert torch.allclose(seq, expected, atol=1e-5)
        assert torch.allclose(direct, expected, atol=1e-5)

    def test_rotation_90_degrees(self):
        """Rotate [1, 0] by 90 degrees -> [0, 1]."""
        source = "(matvec (mat (vec cos_t (- 0 sin_t)) (vec sin_t cos_t)) v)"
        cos_t = torch.tensor(0.0)
        sin_t = torch.tensor(1.0)
        v = torch.tensor([1.0, 0.0])
        inputs_val = {"cos_t": cos_t, "sin_t": sin_t, "v": v}

        seq, direct = _eval_both(source, inputs_val)

        expected = torch.tensor([0.0, 1.0])
        assert torch.allclose(seq, expected, atol=1e-5)
        assert torch.allclose(direct, expected, atol=1e-5)

    def test_stress_strain(self):
        """Stress = C * epsilon (stiffness matrix times strain vector).

        Scheme: (matvec C eps)
        """
        source = "(matvec C eps)"
        # Simple 3x3 stiffness matrix (isotropic-like)
        C = torch.tensor([
            [10.0, 3.0, 0.0],
            [3.0, 10.0, 0.0],
            [0.0, 0.0, 4.0],
        ])
        eps = torch.tensor([0.01, 0.02, 0.005])
        inputs_val = {"C": C, "eps": eps}

        seq, direct = _eval_both(source, inputs_val)

        expected = C @ eps
        assert torch.allclose(seq, expected, atol=1e-5)
        assert torch.allclose(direct, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# 3. TestComposition
# ---------------------------------------------------------------------------

class TestComposition:
    """Chained and nested vector/matrix operations."""

    def test_chained_vector_ops(self):
        """Normalize, scale, then dot with another vector.

        Scheme: (let ((n (normalize v)))
                  (let ((s (scale alpha n)))
                    (dot s w)))
        """
        source = """
        (let ((n (normalize v)))
          (let ((s (scale alpha n)))
            (dot s w)))
        """
        v = torch.tensor([3.0, 4.0, 0.0])
        w = torch.tensor([1.0, 0.0, 0.0])
        alpha = torch.tensor(10.0)
        inputs_val = {"v": v, "w": w, "alpha": alpha}

        graph = compile_scheme(source, inputs={"v": None, "w": None, "alpha": None})
        seq_result = evaluate(graph, inputs_val)
        model = DirectModule(graph)
        direct_result = model({"v": v, "w": w, "alpha": alpha})

        # normalize([3,4,0]) = [0.6, 0.8, 0], scale by 10 = [6, 8, 0]
        # dot with [1,0,0] = 6
        expected = 6.0
        assert seq_result == pytest.approx(expected, rel=1e-4)
        assert direct_result.item() == pytest.approx(expected, rel=1e-4)

    def test_nested_matvec(self):
        """Two matrix-vector products: (matvec A (matvec B v)).

        Equivalent to A @ B @ v.
        """
        source = "(matvec A (matvec B v))"
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        B = torch.tensor([[0.0, 1.0], [1.0, 0.0]])  # permutation matrix
        v = torch.tensor([5.0, 7.0])
        inputs_val = {"A": A, "B": B, "v": v}

        seq, direct = _eval_both(source, inputs_val)

        # B @ v = [7, 5], A @ [7, 5] = [1*7+2*5, 3*7+4*5] = [17, 41]
        expected = A @ (B @ v)
        assert torch.allclose(seq, expected, atol=1e-4)
        assert torch.allclose(direct, expected, atol=1e-4)

    def test_normalize_then_cross(self):
        """Normalize two vectors then take their cross product.

        Scheme: (cross (normalize a) (normalize b))
        """
        source = """
        (let ((na (normalize a))
              (nb (normalize b)))
          (cross na nb))
        """
        a = torch.tensor([2.0, 0.0, 0.0])
        b = torch.tensor([0.0, 3.0, 0.0])
        inputs_val = {"a": a, "b": b}

        seq, direct = _eval_both(source, inputs_val)

        # normalize([2,0,0]) = [1,0,0], normalize([0,3,0]) = [0,1,0]
        # cross([1,0,0], [0,1,0]) = [0,0,1]
        expected = torch.tensor([0.0, 0.0, 1.0])
        assert torch.allclose(seq, expected, atol=1e-5)
        assert torch.allclose(direct, expected, atol=1e-5)

    def test_vector_addition_chain(self):
        """Chain of vector additions: a + b + c.

        Scheme: (+ (+ a b) c)
        """
        source = "(+ (+ a b) c)"
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0, 6.0])
        c = torch.tensor([7.0, 8.0, 9.0])
        inputs_val = {"a": a, "b": b, "c": c}

        seq, direct = _eval_both(source, inputs_val)

        expected = a + b + c
        assert torch.allclose(seq, expected, atol=1e-5)
        assert torch.allclose(direct, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# 4. TestHeatEquationStep
# ---------------------------------------------------------------------------

class TestHeatEquationStep:
    """Forward Euler step of the 1D heat equation.

    u_new = u + alpha * dt * L @ u

    where L is a tridiagonal Laplacian matrix.

    Scheme: (+ u (scale (* alpha dt) (matvec L u)))
    """

    def test_heat_step_uniform(self):
        """Uniform temperature distribution should stay uniform."""
        source = "(+ u (scale (* alpha dt) (matvec L u)))"
        n = 4
        # Tridiagonal Laplacian: L[i,i] = -2, L[i,i+1] = L[i,i-1] = 1
        L = torch.zeros(n, n)
        for i in range(n):
            L[i, i] = -2.0
            if i > 0:
                L[i, i - 1] = 1.0
            if i < n - 1:
                L[i, i + 1] = 1.0

        u = torch.ones(n)  # uniform
        alpha = torch.tensor(0.1)
        dt = torch.tensor(0.01)
        inputs_val = {"u": u, "L": L, "alpha": alpha, "dt": dt}

        seq, direct = _eval_both(source, inputs_val)

        # L @ ones = [-1, 0, 0, -1] (boundary effects)
        expected = u + alpha * dt * (L @ u)
        assert torch.allclose(seq, expected, atol=1e-5)
        assert torch.allclose(direct, expected, atol=1e-5)

    def test_heat_step_bump(self):
        """A temperature bump should diffuse."""
        source = "(+ u (scale (* alpha dt) (matvec L u)))"
        n = 5
        L = torch.zeros(n, n)
        for i in range(n):
            L[i, i] = -2.0
            if i > 0:
                L[i, i - 1] = 1.0
            if i < n - 1:
                L[i, i + 1] = 1.0

        u = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0])
        alpha = torch.tensor(1.0)
        dt = torch.tensor(0.1)
        inputs_val = {"u": u, "L": L, "alpha": alpha, "dt": dt}

        seq, direct = _eval_both(source, inputs_val)

        # Manual: L @ u = [0, 1, -2, 1, 0]
        # u_new = [0, 0, 1, 0, 0] + 0.1 * [0, 1, -2, 1, 0]
        #       = [0, 0.1, 0.8, 0.1, 0]
        expected_np = np.array([0.0, 0.1, 0.8, 0.1, 0.0])
        expected = torch.tensor(expected_np, dtype=torch.float32)

        assert torch.allclose(seq, expected, atol=1e-5), f"seq {seq} != {expected}"
        assert torch.allclose(direct, expected, atol=1e-5), f"direct {direct} != {expected}"

    def test_heat_step_matches_numpy(self):
        """Compare compiled heat step against a numpy reference implementation."""
        source = "(+ u (scale (* alpha dt) (matvec L u)))"
        n = 8
        L_np = np.zeros((n, n))
        for i in range(n):
            L_np[i, i] = -2.0
            if i > 0:
                L_np[i, i - 1] = 1.0
            if i < n - 1:
                L_np[i, i + 1] = 1.0

        np.random.seed(42)
        u_np = np.random.rand(n).astype(np.float32)
        alpha_val = 0.5
        dt_val = 0.01

        L = torch.tensor(L_np, dtype=torch.float32)
        u = torch.tensor(u_np, dtype=torch.float32)
        alpha = torch.tensor(alpha_val)
        dt = torch.tensor(dt_val)
        inputs_val = {"u": u, "L": L, "alpha": alpha, "dt": dt}

        seq, direct = _eval_both(source, inputs_val)

        expected_np = u_np + alpha_val * dt_val * (L_np @ u_np)
        expected = torch.tensor(expected_np, dtype=torch.float32)

        assert torch.allclose(seq, expected, atol=1e-5)
        assert torch.allclose(direct, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# 5. TestLoopWithVectors
# ---------------------------------------------------------------------------

class TestLoopWithVectors:
    """Loops that iterate over vector state."""

    def test_vector_accumulation(self):
        """Iteratively add a constant vector 3 times.

        Scheme: (loop ((v [1 0 0]) (i 0))
                  (if (= i 3) v
                    (recur (+ v [0.1 0.2 0.3]) (+ i 1))))

        After 3 iterations: [1 + 0.3, 0 + 0.6, 0 + 0.9] = [1.3, 0.6, 0.9]
        """
        source = """
        (loop ((v [1 0 0]) (i 0))
          (if (= i 3) v
            (recur (+ v [0.1 0.2 0.3]) (+ i 1))))
        """
        graph = compile_scheme(source)
        seq_result = evaluate(graph, {})
        model = DirectModule(graph)
        direct_result = model({})

        expected = torch.tensor([1.3, 0.6, 0.9])
        if isinstance(seq_result, torch.Tensor):
            assert torch.allclose(seq_result, expected, atol=1e-4), \
                f"seq {seq_result} != {expected}"
        else:
            assert seq_result == pytest.approx(expected.tolist(), rel=1e-4)

        assert torch.allclose(direct_result, expected, atol=1e-4), \
            f"direct {direct_result} != {expected}"

    def test_vector_scaling_loop(self):
        """Scale a vector down by half in each iteration, 4 times.

        Scheme: (loop ((v [8 4 2]) (i 0))
                  (if (= i 4) v
                    (recur (scale 0.5 v) (+ i 1))))

        After 4 iterations: [8, 4, 2] * 0.5^4 = [0.5, 0.25, 0.125]
        """
        source = """
        (loop ((v [8 4 2]) (i 0))
          (if (= i 4) v
            (recur (scale 0.5 v) (+ i 1))))
        """
        graph = compile_scheme(source)
        seq_result = evaluate(graph, {})
        model = DirectModule(graph)
        direct_result = model({})

        expected = torch.tensor([0.5, 0.25, 0.125])
        if isinstance(seq_result, torch.Tensor):
            assert torch.allclose(seq_result, expected, atol=1e-4)

        assert torch.allclose(direct_result, expected, atol=1e-4)

    def test_loop_with_vector_input(self):
        """Loop over vector state with external input for step count.

        Scheme: (loop ((v start) (i 0))
                  (if (= i n) v
                    (recur (+ v delta) (+ i 1))))
        """
        source = """
        (loop ((v start) (i 0))
          (if (= i n) v
            (recur (+ v delta) (+ i 1))))
        """
        start = torch.tensor([0.0, 0.0])
        delta = torch.tensor([1.0, 2.0])
        n = torch.tensor(5.0)
        inputs_val = {"start": start, "delta": delta, "n": n}

        seq, direct = _eval_both(source, inputs_val)

        # After 5 steps: [5.0, 10.0]
        expected = torch.tensor([5.0, 10.0])
        assert torch.allclose(seq, expected, atol=1e-4)
        assert torch.allclose(direct, expected, atol=1e-4)

    def test_iterative_averaging(self):
        """Move a vector toward a target by averaging: v = (v + target) / 2.

        After k iterations, v = target - (target - v0) / 2^k.

        Scheme: (loop ((v [0 0 0]) (i 0))
                  (if (= i 5) v
                    (recur (scale 0.5 (+ v [10 20 30])) (+ i 1))))
        """
        source = """
        (loop ((v [0 0 0]) (i 0))
          (if (= i 5) v
            (recur (scale 0.5 (+ v [10 20 30])) (+ i 1))))
        """
        graph = compile_scheme(source)
        seq_result = evaluate(graph, {})
        model = DirectModule(graph)
        direct_result = model({})

        # Compute expected: v0 = [0,0,0], v_{k+1} = (v_k + t) / 2
        # v1 = [5, 10, 15], v2 = [7.5, 15, 22.5], etc.
        v = np.array([0.0, 0.0, 0.0])
        t = np.array([10.0, 20.0, 30.0])
        for _ in range(5):
            v = 0.5 * (v + t)
        expected = torch.tensor(v, dtype=torch.float32)

        if isinstance(seq_result, torch.Tensor):
            assert torch.allclose(seq_result, expected, atol=1e-3)
        assert torch.allclose(direct_result, expected, atol=1e-3)


# ---------------------------------------------------------------------------
# 6. TestBatchVectorPrograms
# ---------------------------------------------------------------------------

class TestBatchVectorPrograms:
    """Batched execution of vector programs.

    forward_batch() natively handles programs where the batch dimension is
    the leading dimension. For vector-output programs that mix scalar and
    vector inputs, we verify by running individual forward() calls over
    multiple inputs, which is the idiomatic pattern for vector programs.
    """

    def test_batch_heat_equation_multiple_conditions(self):
        """Multiple initial conditions through a heat equation step.

        Each initial condition is run individually via forward() and results
        are collected and compared against numpy reference.
        """
        source = "(+ u (scale (* alpha dt) (matvec L u)))"
        n = 4

        L_np = np.zeros((n, n))
        for i in range(n):
            L_np[i, i] = -2.0
            if i > 0:
                L_np[i, i - 1] = 1.0
            if i < n - 1:
                L_np[i, i + 1] = 1.0
        L = torch.tensor(L_np, dtype=torch.float32)

        # Three different initial conditions
        conditions = [
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            torch.tensor([0.0, 0.0, 1.0, 0.0]),
            torch.tensor([1.0, 1.0, 1.0, 1.0]),
        ]
        alpha = torch.tensor(0.5)
        dt = torch.tensor(0.1)

        graph = compile_scheme(source, inputs={"u": None, "L": None, "alpha": None, "dt": None})
        model = DirectModule(graph)

        for u in conditions:
            result = model({"u": u, "L": L, "alpha": alpha, "dt": dt})
            expected = u + alpha * dt * (L @ u)
            assert torch.allclose(result, expected, atol=1e-5), \
                f"u={u}: result {result} != expected {expected}"

    def test_batch_gravitational_force_multiple_positions(self):
        """Gravitational force for different position vectors.

        F = -G / |r|^3 * r evaluated individually for each r.
        """
        source = "(scale (/ (- 0 G) (pow (norm r) 3)) r)"

        positions = [
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor([0.0, 2.0, 0.0]),
            torch.tensor([3.0, 4.0, 0.0]),
            torch.tensor([1.0, 1.0, 1.0]),
        ]
        G = torch.tensor(1.0)

        graph = compile_scheme(source, inputs={"G": None, "r": None})
        model = DirectModule(graph)

        for r in positions:
            result = model({"G": G, "r": r})
            norm_r = torch.norm(r)
            expected = (-G / norm_r**3) * r
            assert torch.allclose(result, expected, atol=1e-5), \
                f"r={r}: result {result} != expected {expected}"

    def test_batch_spring_force_multiple(self):
        """Spring force for multiple displacement/stiffness combinations."""
        source = "(scale (- 0 k) r)"

        cases = [
            (torch.tensor(1.0), torch.tensor([1.0, 0.0, 0.0])),
            (torch.tensor(2.0), torch.tensor([0.0, 1.0, 0.0])),
            (torch.tensor(3.0), torch.tensor([1.0, 1.0, 1.0])),
        ]

        graph = compile_scheme(source, inputs={"k": None, "r": None})
        model = DirectModule(graph)

        for k, r in cases:
            result = model({"k": k, "r": r})
            expected = -k * r
            assert torch.allclose(result, expected, atol=1e-5)

    def test_batch_kinetic_energy(self):
        """Batch of kinetic energy calculations -- scalar output from vector input.

        dot(v, v) reduces to a scalar per batch element, so forward_batch
        works naturally when v has shape [batch, dim].
        """
        source = "(* 0.5 (dot v v))"

        v_batch = torch.tensor([
            [3.0, 4.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ])

        graph = compile_scheme(source, inputs={"v": None})
        model = DirectModule(graph)

        result = model.forward_batch({"v": v_batch})

        for i in range(3):
            expected = 0.5 * torch.dot(v_batch[i], v_batch[i])
            assert result[i].item() == pytest.approx(expected.item(), rel=1e-4)

    def test_batch_dot_products(self):
        """Batch of dot products: scalar output from two vector inputs."""
        source = "(dot a b)"

        a_batch = torch.tensor([
            [1.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [2.0, 2.0, 2.0],
        ])
        b_batch = torch.tensor([
            [1.0, 1.0, 1.0],
            [4.0, 5.0, 6.0],
            [3.0, 3.0, 3.0],
        ])

        graph = compile_scheme(source, inputs={"a": None, "b": None})
        model = DirectModule(graph)

        result = model.forward_batch({"a": a_batch, "b": b_batch})

        for i in range(3):
            expected = torch.dot(a_batch[i], b_batch[i])
            assert result[i].item() == pytest.approx(expected.item(), rel=1e-4)

    def test_batch_norms(self):
        """Batch of norm calculations: scalar output from vector input."""
        source = "(norm v)"

        v_batch = torch.tensor([
            [3.0, 4.0],
            [1.0, 0.0],
            [5.0, 12.0],
        ])

        graph = compile_scheme(source, inputs={"v": None})
        model = DirectModule(graph)

        result = model.forward_batch({"v": v_batch})

        expected = torch.tensor([5.0, 1.0, 13.0])
        assert torch.allclose(result, expected, atol=1e-4)


# ---------------------------------------------------------------------------
# Additional vector/matrix correctness tests
# ---------------------------------------------------------------------------

class TestVectorOpsEndToEnd:
    """Additional end-to-end tests for individual vector operations."""

    def test_norm_3d(self):
        """norm([3, 4, 0]) = 5."""
        source = "(norm v)"
        v = torch.tensor([3.0, 4.0, 0.0])
        graph = compile_scheme(source, inputs={"v": None})
        result = evaluate(graph, {"v": v})
        assert result == pytest.approx(5.0, rel=1e-5)

    def test_normalize_unit(self):
        """normalize([3, 4]) should have unit length."""
        source = "(normalize v)"
        v = torch.tensor([3.0, 4.0])
        graph = compile_scheme(source, inputs={"v": None})
        result = evaluate(graph, {"v": v})
        assert torch.allclose(result, torch.tensor([0.6, 0.8]), atol=1e-5)

    def test_cross_product_orthogonal(self):
        """cross([1,0,0], [0,1,0]) = [0,0,1]."""
        source = "(cross a b)"
        a = torch.tensor([1.0, 0.0, 0.0])
        b = torch.tensor([0.0, 1.0, 0.0])
        graph = compile_scheme(source, inputs={"a": None, "b": None})
        result = evaluate(graph, {"a": a, "b": b})
        expected = torch.tensor([0.0, 0.0, 1.0])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_outer_product(self):
        """outer([1, 2], [3, 4]) = [[3, 4], [6, 8]]."""
        source = "(outer a b)"
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([3.0, 4.0])
        graph = compile_scheme(source, inputs={"a": None, "b": None})
        result = evaluate(graph, {"a": a, "b": b})
        expected = torch.tensor([[3.0, 4.0], [6.0, 8.0]])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_matrix_transpose(self):
        """transpose of a 2x3 matrix."""
        source = "(transpose M)"
        M = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        graph = compile_scheme(source, inputs={"M": None})
        result = evaluate(graph, {"M": M})
        expected = M.T
        assert torch.allclose(result, expected, atol=1e-5)

    def test_matrix_determinant(self):
        """det of a 2x2 matrix."""
        source = "(det M)"
        M = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        graph = compile_scheme(source, inputs={"M": None})
        result = evaluate(graph, {"M": M})
        expected = 1.0 * 4.0 - 2.0 * 3.0  # = -2.0
        assert result == pytest.approx(expected, abs=1e-4)

    def test_vec_literal_and_ops(self):
        """Vector literal combined with operations.

        Scheme: (dot [1 2 3] [4 5 6]) = 4 + 10 + 18 = 32
        """
        source = "(dot [1 2 3] [4 5 6])"
        graph = compile_scheme(source)
        result = evaluate(graph, {})
        assert result == pytest.approx(32.0, rel=1e-5)

    def test_mat_literal_matvec(self):
        """Matrix literal with matvec.

        (matvec (mat [1 0] [0 1]) v) should be identity.
        """
        source = "(matvec (mat [1 0] [0 1]) v)"
        v = torch.tensor([3.0, 7.0])
        graph = compile_scheme(source, inputs={"v": None})
        result = evaluate(graph, {"v": v})
        assert torch.allclose(result, v, atol=1e-5)


class TestDirectModuleVectors:
    """Verify DirectModule produces correct results for vector programs."""

    def test_direct_matches_sequential_gravitational(self):
        """DirectModule matches sequential evaluator for gravitational force."""
        source = "(scale (/ (- 0 G) (pow (norm r) 3)) r)"
        r = torch.tensor([1.0, 2.0, 2.0])
        G = torch.tensor(10.0)
        inputs_val = {"G": G, "r": r}
        inputs_decl = {"G": None, "r": None}

        graph = compile_scheme(source, inputs=inputs_decl)
        seq_result = evaluate(graph, inputs_val)

        model = DirectModule(graph)
        direct_result = model({"G": G, "r": r})

        assert torch.allclose(seq_result, direct_result, atol=1e-5)

    def test_direct_matches_sequential_matvec(self):
        """DirectModule matches sequential evaluator for matvec."""
        source = "(matvec M v)"
        M = torch.tensor([[2.0, 1.0], [0.0, 3.0]])
        v = torch.tensor([4.0, 5.0])
        inputs_val = {"M": M, "v": v}
        inputs_decl = {"M": None, "v": None}

        graph = compile_scheme(source, inputs=inputs_decl)
        seq_result = evaluate(graph, inputs_val)

        model = DirectModule(graph)
        direct_result = model({"M": M, "v": v})

        expected = torch.tensor([13.0, 15.0])
        assert torch.allclose(seq_result, expected, atol=1e-5)
        assert torch.allclose(direct_result, expected, atol=1e-5)

############################################################
#
# Neural Compiler: Compiling scheme into composable and
#                  differentiable neural network representations
#
# test_vector_ops.py: Unit tests for vector and matrix operations including batched execution
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Unit tests for vector and matrix operations.

Tests all vector construction, vector ops, matrix construction, matrix ops,
broadcasting, conditional branches with vectors, vector inputs, and batched
execution via DirectModule.
"""

import pytest
import torch
from neural_compiler.compiler import compile_scheme
from neural_compiler.evaluator import DirectModule, evaluate
from neural_compiler.parser import parse
from neural_compiler.anf import to_anf
from neural_compiler.graph import build_graph


def _eval(source, inputs=None):
    """Compile and evaluate via the sequential engine (evaluate())."""
    inputs = inputs or {}
    input_decl = {k: None for k in inputs}
    anf = to_anf(parse(source))
    graph = build_graph(anf, inputs=input_decl)
    return evaluate(graph, inputs)


def _direct_eval(source, inputs=None):
    """Compile and evaluate via DirectModule.forward()."""
    inputs = inputs or {}
    input_decl = {k: None for k in inputs}
    graph = compile_scheme(source, inputs=input_decl)
    model = DirectModule(graph)
    tensor_inputs = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            tensor_inputs[k] = v
        else:
            tensor_inputs[k] = torch.tensor(v, dtype=torch.float32)
    return model(tensor_inputs)


# ---------------------------------------------------------------------------
# 1. Vector construction
# ---------------------------------------------------------------------------

class TestVecConstruction:
    """Test (vec ...) and bracket syntax for vector literals."""

    def test_vec_basic(self):
        result = _eval("(vec 1 2 3)")
        expected = torch.tensor([1.0, 2.0, 3.0])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_bracket_syntax(self):
        result = _eval("[1 2 3]")
        expected = torch.tensor([1.0, 2.0, 3.0])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_vec_two_elements(self):
        result = _eval("(vec 4 5)")
        expected = torch.tensor([4.0, 5.0])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_bracket_nested_in_expression(self):
        result = _eval("(dot [1 2 3] [4 5 6])")
        assert result == pytest.approx(32.0, abs=1e-5)

    def test_vec_direct_module(self):
        result = _direct_eval("[1 2 3]")
        expected = torch.tensor([1.0, 2.0, 3.0])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_vec_single_element(self):
        result = _eval("[42]")
        expected = torch.tensor([42.0])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_vec_negative_elements(self):
        result = _eval("(vec (- 0 1) 0 1)")
        expected = torch.tensor([-1.0, 0.0, 1.0])
        assert torch.allclose(result, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# 2. Vector operations
# ---------------------------------------------------------------------------

class TestVecOps:
    """Test vector operations: dot, cross, norm, normalize, vsum, vlen, scale, ref."""

    def test_dot(self):
        result = _eval("(dot [1 2 3] [4 5 6])")
        assert result == pytest.approx(32.0, abs=1e-5)

    def test_dot_direct(self):
        result = _direct_eval("(dot [1 2 3] [4 5 6])")
        assert result.item() == pytest.approx(32.0, abs=1e-5)

    def test_cross(self):
        result = _eval("(cross [1 0 0] [0 1 0])")
        expected = torch.tensor([0.0, 0.0, 1.0])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_cross_direct(self):
        result = _direct_eval("(cross [1 0 0] [0 1 0])")
        expected = torch.tensor([0.0, 0.0, 1.0])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_cross_anticommutative(self):
        result = _eval("(cross [0 1 0] [1 0 0])")
        expected = torch.tensor([0.0, 0.0, -1.0])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_norm(self):
        result = _eval("(norm [3 4])")
        assert result == pytest.approx(5.0, abs=1e-5)

    def test_norm_direct(self):
        result = _direct_eval("(norm [3 4])")
        assert result.item() == pytest.approx(5.0, abs=1e-5)

    def test_norm_3d(self):
        result = _eval("(norm [1 2 2])")
        assert result == pytest.approx(3.0, abs=1e-5)

    def test_normalize(self):
        result = _eval("(normalize [3 4])")
        expected = torch.tensor([0.6, 0.8])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_normalize_direct(self):
        result = _direct_eval("(normalize [3 4])")
        expected = torch.tensor([0.6, 0.8])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_normalize_unit_norm(self):
        result = _eval("(norm (normalize [3 4]))")
        assert result == pytest.approx(1.0, abs=1e-5)

    def test_vsum(self):
        result = _eval("(vsum [1 2 3])")
        assert result == pytest.approx(6.0, abs=1e-5)

    def test_vsum_direct(self):
        result = _direct_eval("(vsum [1 2 3])")
        assert result.item() == pytest.approx(6.0, abs=1e-5)

    def test_vlen(self):
        result = _eval("(vlen [1 2 3])")
        assert result == pytest.approx(3.0, abs=1e-5)

    def test_vlen_direct(self):
        result = _direct_eval("(vlen [1 2 3])")
        assert result.item() == pytest.approx(3.0, abs=1e-5)

    def test_vlen_five(self):
        result = _eval("(vlen [10 20 30 40 50])")
        assert result == pytest.approx(5.0, abs=1e-5)

    def test_scale(self):
        result = _eval("(scale 2 [1 2 3])")
        expected = torch.tensor([2.0, 4.0, 6.0])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_scale_direct(self):
        result = _direct_eval("(scale 2 [1 2 3])")
        expected = torch.tensor([2.0, 4.0, 6.0])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_scale_by_zero(self):
        result = _eval("(scale 0 [1 2 3])")
        expected = torch.tensor([0.0, 0.0, 0.0])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_ref_first(self):
        result = _eval("(ref [10 20 30] 0)")
        assert result == pytest.approx(10.0, abs=1e-5)

    def test_ref_last(self):
        result = _eval("(ref [10 20 30] 2)")
        assert result == pytest.approx(30.0, abs=1e-5)

    def test_ref_middle(self):
        result = _eval("(ref [10 20 30] 1)")
        assert result == pytest.approx(20.0, abs=1e-5)

    def test_ref_direct(self):
        result = _direct_eval("(ref [10 20 30] 0)")
        assert result.item() == pytest.approx(10.0, abs=1e-5)


# ---------------------------------------------------------------------------
# 3. Matrix construction
# ---------------------------------------------------------------------------

class TestMatConstruction:
    """Test (mat ...) for matrix literals."""

    def test_mat_identity_2x2(self):
        result = _eval("(mat [1 0] [0 1])")
        expected = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_mat_general_2x2(self):
        result = _eval("(mat [1 2] [3 4])")
        expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_mat_3x3(self):
        result = _eval("(mat [1 0 0] [0 1 0] [0 0 1])")
        expected = torch.eye(3)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_mat_direct(self):
        result = _direct_eval("(mat [1 0] [0 1])")
        expected = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        assert torch.allclose(result, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# 4. Matrix operations
# ---------------------------------------------------------------------------

class TestMatOps:
    """Test matrix operations: matvec, matmul, transpose, trace, det, inv, outer, eye, zeros, ones."""

    def test_matvec_identity(self):
        result = _eval("(matvec (mat [1 0] [0 1]) [3 4])")
        expected = torch.tensor([3.0, 4.0])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_matvec_scaling(self):
        result = _eval("(matvec (mat [2 0] [0 3]) [1 1])")
        expected = torch.tensor([2.0, 3.0])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_matvec_direct(self):
        result = _direct_eval("(matvec (mat [2 0] [0 3]) [1 1])")
        expected = torch.tensor([2.0, 3.0])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_matmul_identity(self):
        result = _eval("(matmul (mat [1 0] [0 1]) (mat [1 2] [3 4]))")
        expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_matmul_general(self):
        result = _eval("(matmul (mat [1 2] [3 4]) (mat [5 6] [7 8]))")
        expected = torch.tensor([[19.0, 22.0], [43.0, 50.0]])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_matmul_direct(self):
        result = _direct_eval("(matmul (mat [1 0] [0 1]) (mat [1 2] [3 4]))")
        expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_transpose(self):
        result = _eval("(transpose (mat [1 2] [3 4]))")
        expected = torch.tensor([[1.0, 3.0], [2.0, 4.0]])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_transpose_direct(self):
        result = _direct_eval("(transpose (mat [1 2] [3 4]))")
        expected = torch.tensor([[1.0, 3.0], [2.0, 4.0]])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_transpose_identity(self):
        result = _eval("(transpose (mat [1 0] [0 1]))")
        expected = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_trace(self):
        result = _eval("(trace (mat [1 0] [0 2]))")
        assert result == pytest.approx(3.0, abs=1e-5)

    def test_trace_direct(self):
        result = _direct_eval("(trace (mat [1 0] [0 2]))")
        assert result.item() == pytest.approx(3.0, abs=1e-5)

    def test_trace_3x3(self):
        result = _eval("(trace (mat [5 0 0] [0 3 0] [0 0 1]))")
        assert result == pytest.approx(9.0, abs=1e-5)

    def test_det(self):
        result = _eval("(det (mat [1 2] [3 4]))")
        assert result == pytest.approx(-2.0, abs=1e-5)

    def test_det_direct(self):
        result = _direct_eval("(det (mat [1 2] [3 4]))")
        assert result.item() == pytest.approx(-2.0, abs=1e-5)

    def test_det_identity(self):
        result = _eval("(det (mat [1 0] [0 1]))")
        assert result == pytest.approx(1.0, abs=1e-5)

    def test_inv(self):
        result = _eval("(inv (mat [1 0] [0 2]))")
        expected = torch.tensor([[1.0, 0.0], [0.0, 0.5]])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_inv_direct(self):
        result = _direct_eval("(inv (mat [1 0] [0 2]))")
        expected = torch.tensor([[1.0, 0.0], [0.0, 0.5]])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_inv_identity(self):
        result = _eval("(inv (mat [1 0] [0 1]))")
        expected = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_outer(self):
        result = _eval("(outer [1 2] [3 4])")
        expected = torch.tensor([[3.0, 4.0], [6.0, 8.0]])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_outer_direct(self):
        result = _direct_eval("(outer [1 2] [3 4])")
        expected = torch.tensor([[3.0, 4.0], [6.0, 8.0]])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_outer_3d(self):
        result = _eval("(outer [1 0 0] [0 1 0])")
        expected = torch.tensor([[0.0, 1.0, 0.0],
                                  [0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0]])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_eye(self):
        result = _eval("(eye 3)")
        expected = torch.eye(3)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_eye_direct(self):
        result = _direct_eval("(eye 3)")
        expected = torch.eye(3)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_eye_2(self):
        result = _eval("(eye 2)")
        expected = torch.eye(2)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_zeros(self):
        result = _eval("(zeros 3)")
        expected = torch.tensor([0.0, 0.0, 0.0])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_zeros_direct(self):
        result = _direct_eval("(zeros 3)")
        expected = torch.tensor([0.0, 0.0, 0.0])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_ones(self):
        result = _eval("(ones 3)")
        expected = torch.tensor([1.0, 1.0, 1.0])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_ones_direct(self):
        result = _direct_eval("(ones 3)")
        expected = torch.tensor([1.0, 1.0, 1.0])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_ones_5(self):
        result = _eval("(ones 5)")
        expected = torch.ones(5)
        assert torch.allclose(result, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# 5. Broadcasting (element-wise ops on vectors)
# ---------------------------------------------------------------------------

class TestBroadcast:
    """Verify element-wise broadcasting works for vector operands."""

    def test_add_vectors(self):
        result = _eval("(+ [1 2 3] [4 5 6])")
        expected = torch.tensor([5.0, 7.0, 9.0])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_add_vectors_direct(self):
        result = _direct_eval("(+ [1 2 3] [4 5 6])")
        expected = torch.tensor([5.0, 7.0, 9.0])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_scalar_times_vector(self):
        result = _eval("(* 2 [1 2 3])")
        expected = torch.tensor([2.0, 4.0, 6.0])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_scalar_times_vector_direct(self):
        result = _direct_eval("(* 2 [1 2 3])")
        expected = torch.tensor([2.0, 4.0, 6.0])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_sin_vector(self):
        import math
        result = _eval("(sin [0 1.5708])")
        expected = torch.tensor([0.0, 1.0])
        assert torch.allclose(result, expected, atol=1e-3)

    def test_sin_vector_direct(self):
        result = _direct_eval("(sin [0 1.5708])")
        expected = torch.tensor([0.0, 1.0])
        assert torch.allclose(result, expected, atol=1e-3)

    def test_sub_vectors(self):
        result = _eval("(- [10 20] [1 2])")
        expected = torch.tensor([9.0, 18.0])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_sub_vectors_direct(self):
        result = _direct_eval("(- [10 20] [1 2])")
        expected = torch.tensor([9.0, 18.0])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_mul_vectors(self):
        result = _eval("(* [2 3] [4 5])")
        expected = torch.tensor([8.0, 15.0])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_div_vectors(self):
        result = _eval("(/ [10 20] [2 5])")
        expected = torch.tensor([5.0, 4.0])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_cos_vector(self):
        result = _eval("(cos [0 3.14159])")
        expected = torch.tensor([1.0, -1.0])
        assert torch.allclose(result, expected, atol=1e-3)

    def test_exp_vector(self):
        result = _eval("(exp [0 1])")
        import math
        expected = torch.tensor([1.0, math.e])
        assert torch.allclose(result, expected, atol=1e-4)


# ---------------------------------------------------------------------------
# 6. IF with vector branches
# ---------------------------------------------------------------------------

class TestVecIf:
    """Test if-expressions where branches return vectors."""

    def test_if_true_vector(self):
        result = _eval("(if #t [1 2] [3 4])")
        expected = torch.tensor([1.0, 2.0])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_if_false_vector(self):
        result = _eval("(if #f [1 2] [3 4])")
        expected = torch.tensor([3.0, 4.0])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_if_true_vector_direct(self):
        result = _direct_eval("(if #t [1 2] [3 4])")
        expected = torch.tensor([1.0, 2.0])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_if_false_vector_direct(self):
        result = _direct_eval("(if #f [1 2] [3 4])")
        expected = torch.tensor([3.0, 4.0])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_if_with_comparison_vector(self):
        result = _eval("(if (> 5 3) [1 0 0] [0 0 1])")
        expected = torch.tensor([1.0, 0.0, 0.0])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_if_selects_computation(self):
        result = _eval("(if #t (+ [1 1] [2 2]) (- [5 5] [1 1]))")
        expected = torch.tensor([3.0, 3.0])
        assert torch.allclose(result, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# 7. Vector inputs
# ---------------------------------------------------------------------------

class TestVecWithInputs:
    """Test passing vector tensors as inputs."""

    def test_dot_with_input_vectors(self):
        v = torch.tensor([1.0, 2.0, 3.0])
        w = torch.tensor([4.0, 5.0, 6.0])
        result = _direct_eval("(dot v w)", {"v": v, "w": w})
        assert result.item() == pytest.approx(32.0, abs=1e-5)

    def test_norm_with_input_vector(self):
        v = torch.tensor([3.0, 4.0])
        result = _direct_eval("(norm v)", {"v": v})
        assert result.item() == pytest.approx(5.0, abs=1e-5)

    def test_add_input_vectors(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([10.0, 20.0, 30.0])
        result = _direct_eval("(+ a b)", {"a": a, "b": b})
        expected = torch.tensor([11.0, 22.0, 33.0])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_scale_input_vector(self):
        v = torch.tensor([1.0, 2.0, 3.0])
        result = _direct_eval("(scale 3 v)", {"v": v})
        expected = torch.tensor([3.0, 6.0, 9.0])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_cross_with_inputs(self):
        a = torch.tensor([1.0, 0.0, 0.0])
        b = torch.tensor([0.0, 1.0, 0.0])
        result = _direct_eval("(cross a b)", {"a": a, "b": b})
        expected = torch.tensor([0.0, 0.0, 1.0])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_matvec_with_inputs(self):
        m = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
        v = torch.tensor([1.0, 1.0])
        result = _direct_eval("(matvec m v)", {"m": m, "v": v})
        expected = torch.tensor([2.0, 3.0])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_sequential_engine_with_input_vectors(self):
        v = torch.tensor([1.0, 2.0, 3.0])
        w = torch.tensor([4.0, 5.0, 6.0])
        result = _eval("(dot v w)", {"v": v, "w": w})
        assert result == pytest.approx(32.0, abs=1e-5)

    def test_normalize_with_input(self):
        v = torch.tensor([3.0, 4.0])
        result = _direct_eval("(normalize v)", {"v": v})
        expected = torch.tensor([0.6, 0.8])
        assert torch.allclose(result, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# 8. DirectModule batch mode with vectors
# ---------------------------------------------------------------------------

class TestDirectModuleBatch:
    """Test batched execution for vector operations via DirectModule.forward_batch()."""

    def test_batch_dot_product(self):
        """Batch of dot products: each sample has its own v and w."""
        graph = compile_scheme("(dot v w)", inputs={"v": None, "w": None})
        model = DirectModule(graph)
        # Batch of 3 samples, each with 3D vectors
        v = torch.tensor([[1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0],
                          [1.0, 2.0, 3.0]])
        w = torch.tensor([[1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0],
                          [4.0, 5.0, 6.0]])
        result = model.forward_batch({"v": v, "w": w})
        expected = torch.tensor([1.0, 1.0, 32.0])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_batch_matvec(self):
        """Batch of matrix-vector products."""
        graph = compile_scheme("(matvec m v)", inputs={"m": None, "v": None})
        model = DirectModule(graph)
        # Batch of 2 samples: 2x2 matrices and 2D vectors
        m = torch.tensor([[[1.0, 0.0], [0.0, 1.0]],
                          [[2.0, 0.0], [0.0, 3.0]]])
        v = torch.tensor([[3.0, 4.0],
                          [1.0, 1.0]])
        result = model.forward_batch({"m": m, "v": v})
        expected = torch.tensor([[3.0, 4.0],
                                  [2.0, 3.0]])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_batch_if_with_vector_branches(self):
        """Batch where IF selects between vector branches based on scalar condition."""
        graph = compile_scheme(
            "(if (> x 0) [1 0] [0 1])",
            inputs={"x": None},
        )
        model = DirectModule(graph)
        x = torch.tensor([1.0, -1.0, 5.0, -3.0])
        result = model.forward_batch({"x": x})
        expected = torch.tensor([[1.0, 0.0],
                                  [0.0, 1.0],
                                  [1.0, 0.0],
                                  [0.0, 1.0]])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_batch_norm(self):
        """Batch of norm computations."""
        graph = compile_scheme("(norm v)", inputs={"v": None})
        model = DirectModule(graph)
        v = torch.tensor([[3.0, 4.0],
                          [5.0, 12.0],
                          [0.0, 1.0]])
        result = model.forward_batch({"v": v})
        expected = torch.tensor([5.0, 13.0, 1.0])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_batch_normalize(self):
        """Batch of normalize computations."""
        graph = compile_scheme("(normalize v)", inputs={"v": None})
        model = DirectModule(graph)
        v = torch.tensor([[3.0, 4.0],
                          [0.0, 5.0]])
        result = model.forward_batch({"v": v})
        expected = torch.tensor([[0.6, 0.8],
                                  [0.0, 1.0]])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_batch_vsum(self):
        """Batch of vsum computations."""
        graph = compile_scheme("(vsum v)", inputs={"v": None})
        model = DirectModule(graph)
        v = torch.tensor([[1.0, 2.0, 3.0],
                          [10.0, 20.0, 30.0]])
        result = model.forward_batch({"v": v})
        expected = torch.tensor([6.0, 60.0])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_batch_cross(self):
        """Batch of cross products."""
        graph = compile_scheme("(cross a b)", inputs={"a": None, "b": None})
        model = DirectModule(graph)
        a = torch.tensor([[1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0]])
        b = torch.tensor([[0.0, 1.0, 0.0],
                          [0.0, 0.0, 1.0]])
        result = model.forward_batch({"a": a, "b": b})
        expected = torch.tensor([[0.0, 0.0, 1.0],
                                  [1.0, 0.0, 0.0]])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_batch_scale(self):
        """Batch of scale operations with scalar factor and vector."""
        graph = compile_scheme("(scale s v)", inputs={"s": None, "v": None})
        model = DirectModule(graph)
        s = torch.tensor([[2.0],
                          [3.0]])
        v = torch.tensor([[1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0]])
        result = model.forward_batch({"s": s, "v": v})
        expected = torch.tensor([[2.0, 4.0, 6.0],
                                  [12.0, 15.0, 18.0]])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_batch_add_vectors(self):
        """Batch element-wise vector addition."""
        graph = compile_scheme("(+ a b)", inputs={"a": None, "b": None})
        model = DirectModule(graph)
        a = torch.tensor([[1.0, 2.0],
                          [3.0, 4.0]])
        b = torch.tensor([[10.0, 20.0],
                          [30.0, 40.0]])
        result = model.forward_batch({"a": a, "b": b})
        expected = torch.tensor([[11.0, 22.0],
                                  [33.0, 44.0]])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_batch_outer(self):
        """Batch of outer products."""
        graph = compile_scheme("(outer a b)", inputs={"a": None, "b": None})
        model = DirectModule(graph)
        a = torch.tensor([[1.0, 2.0],
                          [3.0, 4.0]])
        b = torch.tensor([[5.0, 6.0],
                          [7.0, 8.0]])
        result = model.forward_batch({"a": a, "b": b})
        expected = torch.tensor([[[5.0, 6.0], [10.0, 12.0]],
                                  [[21.0, 24.0], [28.0, 32.0]]])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_batch_dot_matches_single(self):
        """Verify batched dot product matches individual forward() calls."""
        graph = compile_scheme("(dot v w)", inputs={"v": None, "w": None})
        model = DirectModule(graph)
        torch.manual_seed(42)
        batch_size = 5
        dim = 3
        v_batch = torch.randn(batch_size, dim)
        w_batch = torch.randn(batch_size, dim)
        batch_result = model.forward_batch({"v": v_batch, "w": w_batch})
        for i in range(batch_size):
            single_result = model({"v": v_batch[i], "w": w_batch[i]})
            assert batch_result[i].item() == pytest.approx(
                single_result.item(), abs=1e-4
            ), f"Mismatch at batch index {i}"

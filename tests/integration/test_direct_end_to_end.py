############################################################
#
# Neural Compiler: Compiling scheme into composable and
#                  differentiable neural network representations
#
# test_direct_end_to_end.py: Integration tests verifying DirectModule matches sequential evaluator
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Integration tests: Direct evaluator matches sequential evaluator on all cases.

These tests verify that DirectModule produces identical results to the
sequential evaluate() engine across the full regression suite.
"""

import pytest
import torch
from neural_compiler.compiler import compile_scheme, run_scheme
from neural_compiler.evaluator import evaluate, SchemeGNN, DirectModule


def _both_eval(source, inputs=None):
    """Evaluate with both engines, return (sequential, pyg) results."""
    inputs = inputs or {}
    input_decl = {k: None for k in inputs}
    graph = compile_scheme(source, inputs=input_decl)

    seq_result = evaluate(graph, inputs)

    model = DirectModule(graph)
    tensor_inputs = {k: torch.tensor(v) for k, v in inputs.items()}
    direct_result = model(tensor_inputs).item()

    return seq_result, direct_result


class TestDirectMatchesSequential:
    """Verify Direct evaluator matches sequential on all regression cases."""

    CASES = [
        ("42", {}),
        ("(+ 1 1)", {}),
        ("(* 0 999)", {}),
        ("(- 0 1)", {}),
        ("(/ 1 3)", {}),
        ("(if #t 1 0)", {}),
        ("(if #f 1 0)", {}),
        ("(let ((x 10)) x)", {}),
        ("(+ x 0)", {"x": 7.0}),
        ("(* x 1)", {"x": 7.0}),
        ("(- x x)", {"x": 7.0}),
        ("(+ (* 3 4) 5)", {}),
        ("(- (* 10 10) (+ 50 50))", {}),
        ("(* (+ 1 2) (+ 3 4))", {}),
        ("(/ (+ 10 20) (- 10 5))", {}),
        ("(+ (* 3 x) (- y 1))", {"x": 4.0, "y": 7.0}),
        ("(if (> 5 3) 10 20)", {}),
        ("(if (< 5 3) 10 20)", {}),
        ("(if (> x 0) x (- 0 x))", {"x": 5.0}),
        ("(if (> x 0) x (- 0 x))", {"x": -5.0}),
        ("(let ((a 3) (b 4)) (+ (* a a) (* b b)))", {}),
        ("(* (+ a b) (- c d))", {"a": 3.0, "b": 4.0, "c": 10.0, "d": 3.0}),
    ]

    @pytest.mark.parametrize("source,inputs", CASES)
    def test_pyg_matches_sequential(self, source, inputs):
        seq, pyg = _both_eval(source, inputs)
        assert seq == pytest.approx(pyg, rel=1e-5), (
            f"Mismatch: seq={seq}, pyg={pyg} for {source}"
        )


class TestDirectComplexExpressions:
    """Complex expressions on the Direct evaluator."""

    def test_celsius_to_fahrenheit(self):
        _, pyg = _both_eval("(+ (* (/ 9 5) c) 32)", {"c": 100.0})
        assert pyg == pytest.approx(212.0)

    def test_quadratic_discriminant(self):
        _, pyg = _both_eval(
            "(- (* b b) (* 4 (* a c)))",
            {"a": 1.0, "b": 5.0, "c": 6.0},
        )
        assert pyg == pytest.approx(1.0)

    def test_clamp(self):
        source = "(if (< x lo) lo (if (> x hi) hi x))"
        _, pyg = _both_eval(source, {"x": 5.0, "lo": 0.0, "hi": 10.0})
        assert pyg == pytest.approx(5.0)
        _, pyg = _both_eval(source, {"x": -5.0, "lo": 0.0, "hi": 10.0})
        assert pyg == pytest.approx(0.0)
        _, pyg = _both_eval(source, {"x": 15.0, "lo": 0.0, "hi": 10.0})
        assert pyg == pytest.approx(10.0)

    def test_nested_let_chain(self):
        source = """
        (let ((x2 (* x x))
              (y2 (* y y)))
          (let ((sum (+ x2 y2)))
            (if (> sum 100) 1 0)))
        """
        _, pyg = _both_eval(source, {"x": 7.0, "y": 8.0})
        assert pyg == pytest.approx(1.0)
        _, pyg = _both_eval(source, {"x": 6.0, "y": 8.0})
        assert pyg == pytest.approx(0.0)


class TestDirectModule:
    """Test DirectModule as a proper nn.Module."""

    def test_is_nn_module(self):
        graph = compile_scheme("(+ 1 2)")
        model = DirectModule(graph)
        assert isinstance(model, torch.nn.Module)

    def test_no_learned_parameters(self):
        graph = compile_scheme("(+ (* 3 x) (- y 1))", inputs={"x": None, "y": None})
        model = DirectModule(graph)
        learned_params = [p for p in model.parameters() if p.requires_grad]
        assert len(learned_params) == 0

    def test_has_buffers_for_constants(self):
        graph = compile_scheme("(+ 1 2)")
        model = DirectModule(graph)
        buffer_names = [name for name, _ in model.named_buffers()]
        assert any("const" in name for name in buffer_names)

    def test_matches_simple_gnn(self):
        """Direct module matches the simple sequential SchemeGNN."""
        source = "(+ (* a b) (- c d))"
        inputs_decl = {"a": None, "b": None, "c": None, "d": None}
        inputs_val = {"a": 3.0, "b": 4.0, "c": 10.0, "d": 3.0}
        graph = compile_scheme(source, inputs=inputs_decl)

        simple_model = SchemeGNN(graph)
        simple_result = simple_model(
            {k: torch.tensor(v) for k, v in inputs_val.items()}
        ).item()

        direct_model = DirectModule(graph)
        direct_result = direct_model(
            {k: torch.tensor(v) for k, v in inputs_val.items()}
        ).item()

        assert simple_result == pytest.approx(direct_result)

    def test_repeated_evaluation(self):
        """Same model, different inputs — verify no state leaks."""
        graph = compile_scheme("(* x x)", inputs={"x": None})
        model = DirectModule(graph)

        r1 = model({"x": torch.tensor(3.0)}).item()
        r2 = model({"x": torch.tensor(5.0)}).item()
        r3 = model({"x": torch.tensor(3.0)}).item()

        assert r1 == 9.0
        assert r2 == 25.0
        assert r3 == 9.0


class TestDirectDevice:
    """Test GPU compatibility (MPS on Apple Silicon, CUDA if available)."""

    @pytest.fixture
    def gpu_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        pytest.skip("No GPU available")

    def test_simple_on_gpu(self, gpu_device):
        graph = compile_scheme("(+ 1 2)")
        model = DirectModule(graph).to(gpu_device)
        result = model({})
        assert result.item() == 3.0

    def test_with_inputs_on_gpu(self, gpu_device):
        graph = compile_scheme("(+ (* 3 x) (- y 1))", inputs={"x": None, "y": None})
        model = DirectModule(graph).to(gpu_device)
        result = model({
            "x": torch.tensor(4.0, device=gpu_device),
            "y": torch.tensor(7.0, device=gpu_device),
        })
        assert result.item() == 18.0

    def test_conditional_on_gpu(self, gpu_device):
        graph = compile_scheme("(if (> x 0) x (- 0 x))", inputs={"x": None})
        model = DirectModule(graph).to(gpu_device)
        r1 = model({"x": torch.tensor(5.0, device=gpu_device)}).item()
        r2 = model({"x": torch.tensor(-5.0, device=gpu_device)}).item()
        assert r1 == 5.0
        assert r2 == 5.0

    def test_result_on_correct_device(self, gpu_device):
        graph = compile_scheme("(* 6 7)")
        model = DirectModule(graph).to(gpu_device)
        result = model({})
        assert result.device.type == gpu_device.type

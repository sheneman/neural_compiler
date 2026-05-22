############################################################
#
# Neural Compiler: Compiling scheme into composable and
#                  differentiable neural network representations
#
# test_end_to_end.py: Integration tests for full Scheme to compiled module pipeline
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Integration tests: end-to-end Scheme source → compiled GNN → result.

These tests exercise the full pipeline and serve as regression tests.
Each test case has a known expected output.
"""

import pytest
import torch
from neural_compiler.compiler import compile_scheme, run_scheme
from neural_compiler.evaluator import evaluate
from neural_compiler.evaluator.gnn_module import SchemeGNN


class TestEndToEndArithmetic:
    """Arithmetic expressions compiled and evaluated."""

    @pytest.mark.parametrize(
        "source,inputs,expected",
        [
            ("(+ 1 2)", {}, 3.0),
            ("(- 10 3)", {}, 7.0),
            ("(* 6 7)", {}, 42.0),
            ("(/ 22 7)", {}, pytest.approx(3.142857, rel=1e-4)),
            ("(+ (* 3 4) 5)", {}, 17.0),
            ("(- (* 10 10) (+ 50 50))", {}, 0.0),
            ("(* (+ 1 2) (+ 3 4))", {}, 21.0),
            ("(/ (+ 10 20) (- 10 5))", {}, 6.0),
        ],
    )
    def test_arithmetic(self, source, inputs, expected):
        assert run_scheme(source, inputs) == expected

    @pytest.mark.parametrize(
        "source,inputs,expected",
        [
            ("(+ x 1)", {"x": 5.0}, 6.0),
            ("(* x x)", {"x": 7.0}, 49.0),
            ("(+ (* 3 x) (- y 1))", {"x": 4.0, "y": 7.0}, 18.0),
            ("(/ (+ a b) 2)", {"a": 10.0, "b": 20.0}, 15.0),
        ],
    )
    def test_with_inputs(self, source, inputs, expected):
        assert run_scheme(source, inputs) == expected


class TestEndToEndConditionals:
    """Conditional expressions."""

    @pytest.mark.parametrize(
        "source,inputs,expected",
        [
            ("(if #t 1 0)", {}, 1.0),
            ("(if #f 1 0)", {}, 0.0),
            ("(if (> 5 3) 10 20)", {}, 10.0),
            ("(if (< 5 3) 10 20)", {}, 20.0),
            ("(if (= x 0) 0 (/ 100 x))", {"x": 5.0}, 20.0),
            # Note: (if (= x 0) 0 (/ 100 x)) with x=0 produces NaN because
            # both branches are evaluated (MUX semantics). This is expected
            # for v1 — short-circuit evaluation requires lazy graph execution.
            ("(if (= x 0) 42 (* x 2))", {"x": 0.0}, 42.0),
        ],
    )
    def test_conditionals(self, source, inputs, expected):
        assert run_scheme(source, inputs) == expected


class TestEndToEndLet:
    """Let expressions."""

    @pytest.mark.parametrize(
        "source,inputs,expected",
        [
            ("(let ((x 5)) (* x x))", {}, 25.0),
            ("(let ((a 3) (b 4)) (+ (* a a) (* b b)))", {}, 25.0),
            (
                "(let ((sum (+ a b))) (/ sum 2))",
                {"a": 10.0, "b": 20.0},
                15.0,
            ),
        ],
    )
    def test_let(self, source, inputs, expected):
        assert run_scheme(source, inputs) == expected


class TestEndToEndComplex:
    """Complex expressions combining multiple features."""

    def test_absolute_value(self):
        source = "(if (> x 0) x (- 0 x))"
        assert run_scheme(source, {"x": 5.0}) == 5.0
        assert run_scheme(source, {"x": -5.0}) == 5.0
        assert run_scheme(source, {"x": 0.0}) == 0.0

    def test_clamp(self):
        source = "(if (< x lo) lo (if (> x hi) hi x))"
        assert run_scheme(source, {"x": 5.0, "lo": 0.0, "hi": 10.0}) == 5.0
        assert run_scheme(source, {"x": -5.0, "lo": 0.0, "hi": 10.0}) == 0.0
        assert run_scheme(source, {"x": 15.0, "lo": 0.0, "hi": 10.0}) == 10.0

    def test_quadratic_formula_discriminant(self):
        source = "(- (* b b) (* 4 (* a c)))"
        assert run_scheme(source, {"a": 1.0, "b": 5.0, "c": 6.0}) == 1.0
        assert run_scheme(source, {"a": 1.0, "b": 4.0, "c": 4.0}) == 0.0
        assert run_scheme(source, {"a": 1.0, "b": 1.0, "c": 1.0}) == -3.0

    def test_celsius_to_fahrenheit(self):
        source = "(+ (* (/ 9 5) c) 32)"
        assert run_scheme(source, {"c": 0.0}) == 32.0
        assert run_scheme(source, {"c": 100.0}) == pytest.approx(212.0)

    def test_pythagorean(self):
        source = "(+ (* a a) (* b b))"
        assert run_scheme(source, {"a": 3.0, "b": 4.0}) == 25.0

    def test_let_with_conditional(self):
        source = """
        (let ((d (- (* b b) (* 4 (* a c)))))
          (if (>= d 0) 1 0))
        """
        assert run_scheme(source, {"a": 1.0, "b": 5.0, "c": 6.0}) == 1.0
        assert run_scheme(source, {"a": 1.0, "b": 1.0, "c": 1.0}) == 0.0

    def test_nested_let_chain(self):
        source = """
        (let ((x2 (* x x))
              (y2 (* y y)))
          (let ((sum (+ x2 y2)))
            (if (> sum 100) 1 0)))
        """
        assert run_scheme(source, {"x": 6.0, "y": 8.0}) == 0.0
        assert run_scheme(source, {"x": 7.0, "y": 8.0}) == 1.0


class TestEndToEndTranscendental:
    """Transcendental operations end-to-end."""

    @pytest.mark.parametrize(
        "source,inputs,expected",
        [
            ("(sin 0)", {}, 0.0),
            ("(sin 1.0)", {}, pytest.approx(0.8415, rel=1e-3)),
            ("(cos 0)", {}, 1.0),
            ("(exp 0)", {}, 1.0),
            ("(exp 1)", {}, pytest.approx(2.7183, rel=1e-3)),
            ("(sqrt 4)", {}, pytest.approx(2.0, abs=1e-5)),
            ("(sqrt 9)", {}, pytest.approx(3.0, abs=1e-5)),
            ("(log 1)", {}, pytest.approx(0.0, abs=1e-5)),
            ("(pow 2 10)", {}, pytest.approx(1024.0, rel=1e-4)),
            ("(pow 3 3)", {}, pytest.approx(27.0, rel=1e-4)),
        ],
    )
    def test_transcendental(self, source, inputs, expected):
        assert run_scheme(source, inputs) == expected

    def test_exp_log_roundtrip(self):
        assert run_scheme("(exp (log 2.0))") == pytest.approx(2.0, rel=1e-4)

    def test_sqrt_of_square(self):
        assert run_scheme("(sqrt (* 3 3))") == pytest.approx(3.0, abs=1e-5)

    def test_kinetic_energy(self):
        source = "(* 0.5 (* m (pow v 2)))"
        result = run_scheme(source, {"m": 2.0, "v": 3.0})
        assert result == pytest.approx(9.0, rel=1e-4)

    def test_sin_cos_identity(self):
        source = "(+ (* (sin x) (sin x)) (* (cos x) (cos x)))"
        assert run_scheme(source, {"x": 1.5}) == pytest.approx(1.0, abs=1e-4)

    def test_harmonic_oscillator(self):
        import math
        source = "(* A (sin (+ (* omega t) phi)))"
        result = run_scheme(source, {"A": 2.0, "omega": 3.14159, "t": 0.5, "phi": 0.0})
        expected = 2.0 * math.sin(3.14159 * 0.5)
        assert result == pytest.approx(expected, rel=1e-3)


class TestGNNModule:
    """Test the SchemeGNN nn.Module wrapper."""

    def test_basic_forward(self):
        graph = compile_scheme("(+ (* 3 x) 1)", inputs={"x": None})
        model = SchemeGNN(graph)
        result = model({"x": torch.tensor(4.0)})
        assert result.item() == 13.0

    def test_module_no_grad(self):
        graph = compile_scheme("(* x x)", inputs={"x": None})
        model = SchemeGNN(graph)
        with torch.no_grad():
            result = model({"x": torch.tensor(5.0)})
        assert result.item() == 25.0

    def test_module_matches_evaluator(self):
        source = "(+ (* a b) (- c d))"
        inputs_decl = {"a": None, "b": None, "c": None, "d": None}
        inputs_val = {"a": 3.0, "b": 4.0, "c": 10.0, "d": 3.0}
        graph = compile_scheme(source, inputs=inputs_decl)

        engine_result = evaluate(graph, inputs_val)

        model = SchemeGNN(graph)
        tensor_inputs = {k: torch.tensor(v) for k, v in inputs_val.items()}
        module_result = model(tensor_inputs).item()

        assert engine_result == module_result

    def test_module_is_nn_module(self):
        graph = compile_scheme("(+ 1 2)")
        model = SchemeGNN(graph)
        assert isinstance(model, torch.nn.Module)

    def test_graph_metadata(self):
        graph = compile_scheme(
            "(+ (* 3 x) (- y 1))", inputs={"x": None, "y": None}
        )
        assert graph.depth() == 2
        assert len(graph.input_names) == 2
        assert graph.root_id is not None


class TestRegressionSuite:
    """Fixed regression tests — if any of these break, something fundamental changed."""

    CASES = [
        ("42", {}, 42.0),
        ("(+ 1 1)", {}, 2.0),
        ("(* 0 999)", {}, 0.0),
        ("(- 0 1)", {}, -1.0),
        ("(/ 1 3)", {}, pytest.approx(0.333333, rel=1e-4)),
        ("(if #t 1 0)", {}, 1.0),
        ("(if #f 1 0)", {}, 0.0),
        ("(let ((x 10)) x)", {}, 10.0),
        ("(+ x 0)", {"x": 7.0}, 7.0),
        ("(* x 1)", {"x": 7.0}, 7.0),
        ("(- x x)", {"x": 7.0}, 0.0),
    ]

    @pytest.mark.parametrize("source,inputs,expected", CASES)
    def test_regression(self, source, inputs, expected):
        assert run_scheme(source, inputs) == expected

############################################################
#
# Neural Compiler: Compiling scheme into composable and
#                  differentiable neural network representations
#
# test_message_passing.py: Unit tests for DirectModule arithmetic operations on scalars and tensors
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Unit tests for DirectModule operations."""

import pytest
import torch
from neural_compiler.compiler import compile_scheme
from neural_compiler.evaluator import DirectModule


def _eval(source, inputs=None):
    inputs = inputs or {}
    input_decl = {k: None for k in inputs}
    graph = compile_scheme(source, inputs=input_decl)
    model = DirectModule(graph)
    tensor_inputs = {k: torch.tensor(v) for k, v in inputs.items()}
    return model(tensor_inputs).item()


class TestArithmeticMP:
    def test_add(self):
        assert _eval("(+ 3 4)") == 7.0

    def test_sub(self):
        assert _eval("(- 10 3)") == 7.0

    def test_mul(self):
        assert _eval("(* 6 7)") == 42.0

    def test_div(self):
        assert _eval("(/ 10 4)") == 2.5

    def test_modulo(self):
        assert _eval("(modulo 7 3)") == 1.0

    def test_min(self):
        assert _eval("(min 3 7)") == 3.0

    def test_max(self):
        assert _eval("(max 3 7)") == 7.0

    def test_abs(self):
        assert _eval("(abs -5)", {}) == 5.0


class TestComparisonMP:
    def test_eq_true(self):
        assert _eval("(= 3 3)") == 1.0

    def test_eq_false(self):
        assert _eval("(= 3 4)") == 0.0

    def test_lt(self):
        assert _eval("(< 3 4)") == 1.0

    def test_gt(self):
        assert _eval("(> 4 3)") == 1.0

    def test_le(self):
        assert _eval("(<= 3 3)") == 1.0

    def test_ge(self):
        assert _eval("(>= 4 3)") == 1.0


class TestLogicMP:
    def test_and_true(self):
        assert _eval("(and 1 1)") == 1.0

    def test_and_false(self):
        assert _eval("(and 1 0)") == 0.0

    def test_or_true(self):
        assert _eval("(or 0 1)") == 1.0

    def test_or_false(self):
        assert _eval("(or 0 0)") == 0.0

    def test_not_true(self):
        assert _eval("(not 1)") == 0.0

    def test_not_false(self):
        assert _eval("(not 0)") == 1.0


class TestTranscendentalMP:
    def test_sin(self):
        assert _eval("(sin 1.0)") == pytest.approx(0.8415, rel=1e-3)

    def test_cos(self):
        assert _eval("(cos 0)") == pytest.approx(1.0, abs=1e-6)

    def test_exp(self):
        assert _eval("(exp 0)") == pytest.approx(1.0, abs=1e-6)

    def test_exp_one(self):
        assert _eval("(exp 1)") == pytest.approx(2.7183, rel=1e-3)

    def test_sqrt(self):
        assert _eval("(sqrt 9)") == pytest.approx(3.0, abs=1e-5)

    def test_log(self):
        assert _eval("(log 1)") == pytest.approx(0.0, abs=1e-5)

    def test_pow(self):
        assert _eval("(pow 2 10)") == pytest.approx(1024.0, rel=1e-4)

    def test_pow_with_input(self):
        assert _eval("(pow x 2)", {"x": 5.0}) == pytest.approx(25.0, rel=1e-4)

    def test_exp_log_roundtrip(self):
        assert _eval("(exp (log 2))") == pytest.approx(2.0, rel=1e-4)

    def test_sin_cos_identity(self):
        result = _eval("(+ (pow (sin x) 2) (pow (cos x) 2))", {"x": 1.5})
        assert result == pytest.approx(1.0, abs=1e-4)


class TestIfMP:
    def test_if_true(self):
        assert _eval("(if #t 10 20)") == 10.0

    def test_if_false(self):
        assert _eval("(if #f 10 20)") == 20.0

    def test_if_with_comparison(self):
        assert _eval("(if (> x 0) x (- 0 x))", {"x": 5.0}) == 5.0
        assert _eval("(if (> x 0) x (- 0 x))", {"x": -5.0}) == 5.0

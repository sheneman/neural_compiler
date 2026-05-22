############################################################
#
# Neural Compiler: Compiling scheme into composable and
#                  differentiable neural network representations
#
# test_ops.py: Unit tests for primitive operations (arithmetic, comparisons, transcendentals)
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Unit tests for primitive operations."""

import pytest
import torch
from neural_compiler.ops.primitives import evaluate_op


def T(x):
    return torch.tensor(x, dtype=torch.float32)


class TestArithmetic:
    def test_add(self):
        assert evaluate_op("+", [T(3.0), T(4.0)]).item() == 7.0

    def test_sub(self):
        assert evaluate_op("-", [T(10.0), T(3.0)]).item() == 7.0

    def test_mul(self):
        assert evaluate_op("*", [T(3.0), T(4.0)]).item() == 12.0

    def test_div(self):
        assert evaluate_op("/", [T(10.0), T(4.0)]).item() == 2.5

    def test_modulo(self):
        assert evaluate_op("modulo", [T(7.0), T(3.0)]).item() == 1.0

    def test_abs_positive(self):
        assert evaluate_op("abs", [T(5.0)]).item() == 5.0

    def test_abs_negative(self):
        assert evaluate_op("abs", [T(-5.0)]).item() == 5.0

    def test_min(self):
        assert evaluate_op("min", [T(3.0), T(7.0)]).item() == 3.0

    def test_max(self):
        assert evaluate_op("max", [T(3.0), T(7.0)]).item() == 7.0


class TestComparison:
    def test_eq_true(self):
        assert evaluate_op("=", [T(3.0), T(3.0)]).item() == 1.0

    def test_eq_false(self):
        assert evaluate_op("=", [T(3.0), T(4.0)]).item() == 0.0

    def test_lt_true(self):
        assert evaluate_op("<", [T(3.0), T(4.0)]).item() == 1.0

    def test_lt_false(self):
        assert evaluate_op("<", [T(4.0), T(3.0)]).item() == 0.0

    def test_gt(self):
        assert evaluate_op(">", [T(4.0), T(3.0)]).item() == 1.0

    def test_le(self):
        assert evaluate_op("<=", [T(3.0), T(3.0)]).item() == 1.0

    def test_ge(self):
        assert evaluate_op(">=", [T(4.0), T(3.0)]).item() == 1.0


class TestLogic:
    def test_not_true(self):
        assert evaluate_op("not", [T(1.0)]).item() == 0.0

    def test_not_false(self):
        assert evaluate_op("not", [T(0.0)]).item() == 1.0

    def test_and_true(self):
        assert evaluate_op("and", [T(1.0), T(1.0)]).item() == 1.0

    def test_and_false(self):
        assert evaluate_op("and", [T(1.0), T(0.0)]).item() == 0.0

    def test_or_true(self):
        assert evaluate_op("or", [T(0.0), T(1.0)]).item() == 1.0

    def test_or_false(self):
        assert evaluate_op("or", [T(0.0), T(0.0)]).item() == 0.0


class TestIf:
    def test_if_true(self):
        assert evaluate_op("if", [T(1.0), T(10.0), T(20.0)]).item() == 10.0

    def test_if_false(self):
        assert evaluate_op("if", [T(0.0), T(10.0), T(20.0)]).item() == 20.0

    def test_if_nonzero_is_truthy(self):
        assert evaluate_op("if", [T(5.0), T(10.0), T(20.0)]).item() == 10.0


class TestTranscendental:
    def test_sin(self):
        assert evaluate_op("sin", [T(0.0)]).item() == pytest.approx(0.0, abs=1e-6)

    def test_sin_pi_half(self):
        import math
        assert evaluate_op("sin", [T(math.pi / 2)]).item() == pytest.approx(1.0, abs=1e-6)

    def test_cos(self):
        assert evaluate_op("cos", [T(0.0)]).item() == pytest.approx(1.0, abs=1e-6)

    def test_cos_pi(self):
        import math
        assert evaluate_op("cos", [T(math.pi)]).item() == pytest.approx(-1.0, abs=1e-5)

    def test_exp(self):
        assert evaluate_op("exp", [T(0.0)]).item() == pytest.approx(1.0, abs=1e-6)

    def test_exp_one(self):
        import math
        assert evaluate_op("exp", [T(1.0)]).item() == pytest.approx(math.e, rel=1e-5)

    def test_sqrt(self):
        assert evaluate_op("sqrt", [T(9.0)]).item() == pytest.approx(3.0, abs=1e-6)

    def test_sqrt_clamped(self):
        result = evaluate_op("sqrt", [T(-1.0)]).item()
        assert result >= 0.0

    def test_log(self):
        import math
        assert evaluate_op("log", [T(math.e)]).item() == pytest.approx(1.0, rel=1e-5)

    def test_log_one(self):
        assert evaluate_op("log", [T(1.0)]).item() == pytest.approx(0.0, abs=1e-6)

    def test_log_clamped(self):
        result = evaluate_op("log", [T(-1.0)]).item()
        assert not torch.isnan(torch.tensor(result))

    def test_pow(self):
        assert evaluate_op("pow", [T(2.0), T(10.0)]).item() == pytest.approx(1024.0, rel=1e-5)

    def test_pow_square(self):
        assert evaluate_op("pow", [T(5.0), T(2.0)]).item() == pytest.approx(25.0, rel=1e-5)

    def test_pow_fractional(self):
        assert evaluate_op("pow", [T(8.0), T(1.0 / 3.0)]).item() == pytest.approx(2.0, rel=1e-4)


class TestErrors:
    def test_unknown_op(self):
        with pytest.raises(ValueError, match="Unknown operation"):
            evaluate_op("bogus", [T(1.0)])

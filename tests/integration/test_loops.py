############################################################
#
# Neural Compiler: Compiling scheme into composable and
#                  differentiable neural network representations
#
# test_loops.py: Integration tests for loop/recur with sequential and DirectModule evaluators
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Integration tests for loop/recur: end-to-end with both evaluators."""

import pytest
import torch
from neural_compiler.compiler import compile_scheme, run_scheme
from neural_compiler.evaluator import evaluate, SchemeGNN, DirectModule


def _both_eval(source, inputs=None):
    """Evaluate with sequential and DirectModule, return both results."""
    inputs = inputs or {}
    input_decl = {k: None for k in inputs}
    graph = compile_scheme(source, inputs=input_decl)

    seq_result = evaluate(graph, inputs)

    model = DirectModule(graph)
    tensor_inputs = {k: torch.tensor(v) for k, v in inputs.items()}
    pyg_result = model(tensor_inputs).item()

    return seq_result, pyg_result


class TestFactorial:
    """Factorial: n! via tail recursion."""

    FACTORIAL_SRC = "(loop ((n N) (acc 1)) (if (= n 0) acc (recur (- n 1) (* acc n))))"

    @pytest.mark.parametrize("n,expected", [
        (0, 1.0),
        (1, 1.0),
        (2, 2.0),
        (3, 6.0),
        (4, 24.0),
        (5, 120.0),
        (6, 720.0),
        (10, 3628800.0),
    ])
    def test_factorial_sequential(self, n, expected):
        assert run_scheme(self.FACTORIAL_SRC, {"N": float(n)}) == expected

    @pytest.mark.parametrize("n,expected", [
        (0, 1.0),
        (1, 1.0),
        (5, 120.0),
        (10, 3628800.0),
    ])
    def test_factorial_both_match(self, n, expected):
        seq, pyg = _both_eval(self.FACTORIAL_SRC, {"N": float(n)})
        assert seq == pytest.approx(expected)
        assert pyg == pytest.approx(expected)


class TestSumToN:
    """Sum of 1 to n."""

    SUM_SRC = "(loop ((i n) (sum 0)) (if (= i 0) sum (recur (- i 1) (+ sum i))))"

    @pytest.mark.parametrize("n,expected", [
        (0, 0.0),
        (1, 1.0),
        (5, 15.0),
        (10, 55.0),
        (100, 5050.0),
    ])
    def test_sum(self, n, expected):
        seq, pyg = _both_eval(self.SUM_SRC, {"n": float(n)})
        assert seq == pytest.approx(expected)
        assert pyg == pytest.approx(expected)


class TestGCD:
    """Greatest common divisor via Euclidean algorithm."""

    GCD_SRC = "(loop ((a A) (b B)) (if (= b 0) a (recur b (modulo a b))))"

    @pytest.mark.parametrize("a,b,expected", [
        (48, 18, 6.0),
        (100, 75, 25.0),
        (17, 13, 1.0),
        (12, 12, 12.0),
        (0, 5, 5.0),
        (7, 0, 7.0),
    ])
    def test_gcd(self, a, b, expected):
        seq, pyg = _both_eval(self.GCD_SRC, {"A": float(a), "B": float(b)})
        assert seq == pytest.approx(expected)
        assert pyg == pytest.approx(expected)


class TestPower:
    """Integer exponentiation via repeated multiplication."""

    POW_SRC = "(loop ((b base) (e exp) (acc 1)) (if (= e 0) acc (recur b (- e 1) (* acc b))))"

    @pytest.mark.parametrize("base,exp,expected", [
        (2, 0, 1.0),
        (2, 1, 2.0),
        (2, 10, 1024.0),
        (3, 4, 81.0),
        (5, 3, 125.0),
    ])
    def test_power(self, base, exp, expected):
        seq, pyg = _both_eval(self.POW_SRC, {"base": float(base), "exp": float(exp)})
        assert seq == pytest.approx(expected)
        assert pyg == pytest.approx(expected)


class TestCountdown:
    """Simple countdown — tests that loop terminates correctly."""

    def test_countdown_to_zero(self):
        source = "(loop ((n 100)) (if (= n 0) n (recur (- n 1))))"
        seq, pyg = _both_eval(source)
        assert seq == 0.0
        assert pyg == 0.0


class TestFibonacci:
    """Fibonacci via tail recursion with two accumulators."""

    FIB_SRC = "(loop ((n N) (a 0) (b 1)) (if (= n 0) a (recur (- n 1) b (+ a b))))"

    @pytest.mark.parametrize("n,expected", [
        (0, 0.0),
        (1, 1.0),
        (2, 1.0),
        (3, 2.0),
        (4, 3.0),
        (5, 5.0),
        (6, 8.0),
        (10, 55.0),
        (20, 6765.0),
    ])
    def test_fibonacci(self, n, expected):
        seq, pyg = _both_eval(self.FIB_SRC, {"N": float(n)})
        assert seq == pytest.approx(expected)
        assert pyg == pytest.approx(expected)


class TestLoopWithOuterLet:
    """Loop with let bindings around it."""

    def test_let_wrapping_loop(self):
        source = """
        (let ((start 10))
          (loop ((n start) (sum 0))
            (if (= n 0) sum (recur (- n 1) (+ sum n)))))
        """
        seq, pyg = _both_eval(source)
        assert seq == pytest.approx(55.0)
        assert pyg == pytest.approx(55.0)


class TestLoopWithInputs:
    """Loop using external inputs."""

    def test_factorial_with_input(self):
        source = "(loop ((n x) (acc 1)) (if (= n 0) acc (recur (- n 1) (* acc n))))"
        seq, pyg = _both_eval(source, {"x": 7.0})
        assert seq == pytest.approx(5040.0)
        assert pyg == pytest.approx(5040.0)


class TestBaseCase:
    """Loops that terminate immediately (base case on first iteration)."""

    def test_immediate_termination(self):
        source = "(loop ((n 0) (acc 1)) (if (= n 0) acc (recur (- n 1) (* acc n))))"
        seq, pyg = _both_eval(source)
        assert seq == 1.0
        assert pyg == 1.0


class TestLoopGPU:
    """GPU tests for loops."""

    @pytest.fixture
    def gpu_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        pytest.skip("No GPU available")

    def test_factorial_gpu(self, gpu_device):
        source = "(loop ((n 5) (acc 1)) (if (= n 0) acc (recur (- n 1) (* acc n))))"
        graph = compile_scheme(source)
        model = DirectModule(graph).to(gpu_device)
        result = model({})
        assert result.item() == 120.0

    def test_loop_with_input_gpu(self, gpu_device):
        source = "(loop ((i n) (sum 0)) (if (= i 0) sum (recur (- i 1) (+ sum i))))"
        graph = compile_scheme(source, inputs={"n": None})
        model = DirectModule(graph).to(gpu_device)
        result = model({"n": torch.tensor(10.0, device=gpu_device)})
        assert result.item() == 55.0


class TestLoopModule:
    """Test SchemeGNN (sequential module) with loops."""

    def test_factorial_module(self):
        source = "(loop ((n 5) (acc 1)) (if (= n 0) acc (recur (- n 1) (* acc n))))"
        graph = compile_scheme(source)
        model = SchemeGNN(graph)
        result = model({})
        assert result.item() == 120.0

    def test_repeated_calls(self):
        source = "(loop ((n x) (acc 1)) (if (= n 0) acc (recur (- n 1) (* acc n))))"
        graph = compile_scheme(source, inputs={"x": None})
        model = SchemeGNN(graph)

        assert model({"x": torch.tensor(3.0)}).item() == 6.0
        assert model({"x": torch.tensor(5.0)}).item() == 120.0
        assert model({"x": torch.tensor(0.0)}).item() == 1.0

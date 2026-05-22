############################################################
#
# Neural Compiler: Compiling scheme into composable and
#                  differentiable neural network representations
#
# compiler.py: Top-level compiler orchestrating the Scheme to AST to ANF to graph to module pipeline
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Top-level compiler: Scheme source → ComputeGraph → evaluation."""

from __future__ import annotations
from neural_compiler.parser import parse
from neural_compiler.anf import to_anf
from neural_compiler.anf.tco import optimize_tco
from neural_compiler.graph import build_graph, ComputeGraph
from neural_compiler.evaluator import evaluate


def compile_scheme(
    source: str,
    inputs: dict[str, None] | None = None,
) -> ComputeGraph:
    """Compile a Scheme expression to a compute graph.

    Args:
        source: Scheme source code (a single expression).
        inputs: Dict of input variable names expected by the expression.

    Returns:
        A ComputeGraph that can be evaluated with concrete input values.

    Example:
        >>> graph = compile_scheme("(+ (* 3 x) (- y 1))", inputs={"x": None, "y": None})
        >>> result = evaluate(graph, {"x": 4.0, "y": 7.0})
        >>> result  # 3*4 + (7-1) = 18.0
        18.0
    """
    ast = parse(source)
    anf = to_anf(ast)
    anf = optimize_tco(anf)
    graph = build_graph(anf, inputs=inputs)
    return graph


def run_scheme(source: str, inputs: dict[str, float] | None = None) -> float:
    """Compile and immediately evaluate a Scheme expression.

    Args:
        source: Scheme source code (a single expression).
        inputs: Dict of input variable names to float values.

    Returns:
        The scalar result of the computation.
    """
    inputs = inputs or {}
    input_decl = {k: None for k in inputs}
    graph = compile_scheme(source, inputs=input_decl)
    return evaluate(graph, inputs)

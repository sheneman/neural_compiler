############################################################
#
# Neural Compiler: Compiling scheme into composable and
#                  differentiable neural network representations
#
# __init__.py: Evaluator module exports for sequential, GNN, and DirectModule backends
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Evaluator backends for compiled compute graphs.

Three execution modes:
  - evaluate(): Sequential Python loop over nodes (educational, CPU-only)
  - SchemeGNN: torch.nn.Module wrapper for sequential evaluation
  - DirectModule: Flat instruction execution (~1-3x overhead)
"""

from neural_compiler.evaluator.engine import evaluate
from neural_compiler.evaluator.gnn_module import SchemeGNN
from neural_compiler.evaluator.direct_module import DirectModule

__all__ = ["evaluate", "SchemeGNN", "DirectModule"]

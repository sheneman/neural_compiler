############################################################
#
# Neural Compiler: Compiling scheme into composable and
#                  differentiable neural network representations
#
# __init__.py: Operations module exports for primitive operation evaluation
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Node operations: fixed-weight implementations of primitive operations."""

from neural_compiler.ops.primitives import evaluate_op

__all__ = ["evaluate_op"]

############################################################
#
# Neural Compiler: Compiling scheme into composable and
#                  differentiable neural network representations
#
# __init__.py: Main package entry point; exports compile_scheme
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Neural Compiler: Scheme to Graph Neural Network compiler."""

from neural_compiler.compiler import compile_scheme

__all__ = ["compile_scheme"]

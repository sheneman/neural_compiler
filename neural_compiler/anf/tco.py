############################################################
#
# Neural Compiler: Compiling scheme into composable and
#                  differentiable neural network representations
#
# tco.py: Tail-call optimization converting self-tail-recursive letrec to iterative loop/recur
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Tail-call optimization: convert self-tail-recursive letrec to loop/recur.

When a letrec binds a single function whose only self-calls are in tail
position, the entire construct can be replaced with a loop/recur — giving
O(1) stack instead of O(depth).

Two cases:
  1. Letrec body is a direct call (f args...) → eliminate letrec, produce ANFLoop
  2. Letrec body uses f indirectly → keep letrec but replace lambda body with
     an internal loop so the function no longer recurses
"""

from __future__ import annotations
from neural_compiler.anf.anf_nodes import (
    ANFNode,
    ANFConst,
    ANFVar,
    ANFLet,
    ANFIf,
    ANFApp,
    ANFLambda,
    ANFLetrec,
    ANFLoop,
    ANFRecur,
)
from neural_compiler.parser.ast_nodes import PRIMITIVES


def optimize_tco(node: ANFNode) -> ANFNode:
    """Walk the ANF tree and optimize eligible letrec nodes to loop/recur."""
    if isinstance(node, ANFLetrec):
        result = _try_tco_letrec(node)
        return _walk(result)
    return _walk(node)


def _walk(node: ANFNode) -> ANFNode:
    """Recursively apply TCO to all sub-expressions."""
    if isinstance(node, (ANFConst, ANFVar)):
        return node

    if isinstance(node, ANFLet):
        return ANFLet(
            name=node.name,
            rhs=_walk(node.rhs),
            body=_walk(node.body),
        )

    if isinstance(node, ANFIf):
        return ANFIf(
            test=node.test,
            then_=_walk(node.then_),
            else_=_walk(node.else_),
        )

    if isinstance(node, ANFApp):
        return node

    if isinstance(node, ANFLambda):
        return ANFLambda(params=node.params, body=_walk(node.body))

    if isinstance(node, ANFLetrec):
        result = _try_tco_letrec(node)
        if not isinstance(result, ANFLetrec):
            return _walk(result)
        new_bindings = tuple(
            (name, ANFLambda(params=lam.params, body=_walk(lam.body)))
            for name, lam in result.bindings
        )
        return ANFLetrec(bindings=new_bindings, body=_walk(result.body))

    if isinstance(node, ANFLoop):
        return ANFLoop(
            params=node.params,
            inits=node.inits,
            body=_walk(node.body),
        )

    if isinstance(node, ANFRecur):
        return node

    return node


def _try_tco_letrec(node: ANFLetrec) -> ANFNode:
    """Try to optimize a letrec. Returns the original node if not eligible."""
    if len(node.bindings) != 1:
        return node

    name, lam = node.bindings[0]

    ok, transformed_body = _replace_tail_calls(name, lam.body, in_tail=True)
    if not ok:
        return node

    if _is_direct_call(name, node.body):
        inits = node.body.args
        return ANFLoop(params=lam.params, inits=inits, body=transformed_body)

    loop_body = ANFLoop(
        params=lam.params,
        inits=tuple(ANFVar(p) for p in lam.params),
        body=transformed_body,
    )
    new_lam = ANFLambda(params=lam.params, body=loop_body)
    return ANFLetrec(
        bindings=((name, new_lam),),
        body=_inline_calls(name, lam.params, node.body),
    )


def _replace_tail_calls(
    name: str, node: ANFNode, in_tail: bool
) -> tuple[bool, ANFNode]:
    """Replace tail self-calls with ANFRecur. Returns (success, new_node).

    Fails if any self-call is in a non-tail position.
    """
    if isinstance(node, ANFApp):
        if isinstance(node.func, ANFVar) and node.func.name == name:
            if not in_tail:
                return False, node
            return True, ANFRecur(args=node.args)
        return True, node

    if isinstance(node, (ANFConst, ANFVar)):
        return True, node

    if isinstance(node, ANFIf):
        if _contains_call_to(name, node.test):
            return False, node
        ok_then, new_then = _replace_tail_calls(name, node.then_, in_tail)
        if not ok_then:
            return False, node
        ok_else, new_else = _replace_tail_calls(name, node.else_, in_tail)
        if not ok_else:
            return False, node
        return True, ANFIf(test=node.test, then_=new_then, else_=new_else)

    if isinstance(node, ANFLet):
        if _contains_call_to(name, node.rhs):
            return False, node
        ok_body, new_body = _replace_tail_calls(name, node.body, in_tail)
        if not ok_body:
            return False, node
        return True, ANFLet(name=node.name, rhs=node.rhs, body=new_body)

    if isinstance(node, ANFRecur):
        return True, node

    return True, node


def _contains_call_to(name: str, node: ANFNode) -> bool:
    """Check if node or any descendant contains a call to `name`."""
    if isinstance(node, ANFApp):
        if isinstance(node.func, ANFVar) and node.func.name == name:
            return True
        return False

    if isinstance(node, (ANFConst, ANFVar)):
        return False

    if isinstance(node, ANFIf):
        return (
            _contains_call_to(name, node.test)
            or _contains_call_to(name, node.then_)
            or _contains_call_to(name, node.else_)
        )

    if isinstance(node, ANFLet):
        return _contains_call_to(name, node.rhs) or _contains_call_to(name, node.body)

    if isinstance(node, ANFLambda):
        return _contains_call_to(name, node.body)

    return False


def _is_direct_call(name: str, node: ANFNode) -> bool:
    """Check if node is a direct application of `name`."""
    return (
        isinstance(node, ANFApp)
        and isinstance(node.func, ANFVar)
        and node.func.name == name
    )


def _inline_calls(
    name: str, params: tuple[str, ...], node: ANFNode
) -> ANFNode:
    """Replace calls to `name` in the letrec body with inline ANFLoop invocations.

    For case 2 (indirect use), each call site (f args...) becomes
    (loop ((params args)) body) — but since the lambda body already
    contains the loop, the call just invokes the non-recursive lambda.
    No transformation needed here — the letrec binding is kept and the
    lambda's body handles the looping.
    """
    return node

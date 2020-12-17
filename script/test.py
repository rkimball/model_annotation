import tvm
import numpy as np
import tvm.ir
from tvm import relay, tir
from tvm.relay.dataflow_pattern import TupleGetItemPattern, is_op, wildcard
from tvm.relay.op.contrib.register import get_pattern_table, register_pattern_table
from tvm.relay.op.annotation import compiler_begin, compiler_end
from tvm.relay.expr import Call, TupleGetItem, Var, Constant, Tuple
from tvm.ir import Op
# from .register import register_pattern_table


@register_pattern_table("test_target")
def pattern_table():
    """Get the APU compiler pattern table."""

    def multiply():
        pattern = is_op("multiply")(wildcard(), wildcard())
        return pattern

    def check_multiply(extract):
        """Check if multiply is supported."""
        return True

    return [
        ("test_target.mul", multiply(), check_multiply)
    ]


# def make_add_pattern():
#     """Create a pattern to match x + y

#        a  b  a  b
#         \/    \/
#         add  add
#          \   /
#           \ /
#           mul
#           /  \
#        c /  c |
#        \/   \/
#        mul  mul
#         \   /
#          \ /
#          add

def get_model():
    a = relay.var("a", shape=(2, 3))
    b = relay.var("b", shape=(2, 3))
    c = relay.var("c", shape=(2, 3))
    add1 = relay.add(a, b)
    add2 = relay.add(a, b)
    mul1 = relay.multiply(add1, add2)
    mul2 = relay.multiply(mul1, c)
    mul3 = relay.multiply(mul1, c)
    add3 = relay.add(mul2, mul3)
    func = relay.Function([a, b, c], add3)

    mod = tvm.IRModule()
    mod["main"] = func
    return mod


def get_annotated_model():
    a = relay.var("a", shape=(2, 3))
    b = relay.var("b", shape=(2, 3))
    c = relay.var("c", shape=(2, 3))
    a_b = compiler_begin(a, "default")
    b_b = compiler_begin(b, "default")
    add1 = relay.add(a_b, b_b)
    add2 = relay.add(a_b, b_b)
    add1_e = compiler_end(add1, "default")
    add2_e = compiler_end(add2, "default")

    add1_b = compiler_begin(add1_e, "test_target")
    add2_b = compiler_begin(add2_e, "test_target")
    mul1 = relay.multiply(add1_b, add2_b)
    c_b = compiler_begin(c, "test_target")
    mul2 = relay.multiply(mul1, c_b)
    mul3 = relay.multiply(mul1, c_b)
    mul2_e = compiler_end(mul2, "test_target")
    mul3_e = compiler_end(mul3, "test_target")

    mul2_b = compiler_begin(mul2_e, "default")
    mul3_b = compiler_begin(mul3_e, "default")
    add3 = relay.add(mul2_b, mul3_b)
    add3_e = compiler_end(add3, "default")
    func = relay.Function([a, b, c], add3_e)

    mod = tvm.IRModule()
    mod["main"] = func
    return mod


def get_annotated2_model():
    a = relay.var("a", shape=(2, 3))
    b = relay.var("b", shape=(2, 3))
    c = relay.var("c", shape=(2, 3))
    a_b = compiler_begin(a, "default")
    b_b = compiler_begin(b, "default")
    add1 = relay.add(a_b, b_b)
    add2 = relay.add(a_b, b_b)
    add1_e = compiler_end(add1, "default")
    add2_e = compiler_end(add2, "default")

    add1_b = compiler_begin(add1_e, "test_target")
    add2_b = compiler_begin(add2_e, "test_target")
    mul1 = relay.multiply(add1_b, add2_b)
    mul1_e = compiler_end(mul1, "test_target")

    c_b = compiler_begin(c, "test_target")
    mul1_b = compiler_begin(mul1_e, "test_target")
    mul2 = relay.multiply(mul1_b, c_b)
    mul3 = relay.multiply(mul1_b, c_b)
    mul2_e = compiler_end(mul2, "test_target")
    mul3_e = compiler_end(mul3, "test_target")

    mul2_b = compiler_begin(mul2_e, "default")
    mul3_b = compiler_begin(mul3_e, "default")
    add3 = relay.add(mul2_b, mul3_b)
    add3_e = compiler_end(add3, "default")
    func = relay.Function([a, b, c], add3_e)

    mod = tvm.IRModule()
    mod["main"] = func
    return mod


def get_placement(expr):
    target_ops = ["multiply"]
    placement = "default"
    if isinstance(expr, Call):
        if isinstance(expr.op, Op):
            if expr.op.name in target_ops:
                placement = "test_target"
    return placement


if __name__ == "__main__":
    mod = get_model()
    print(mod)

    mod = relay.transform.AnnotateCompiler(get_placement)(mod)

    print(mod)

    mod = relay.transform.MergeCompilerRegions()(mod)
    mod = relay.transform.PartitionGraph()(mod)

    print(mod)

    # ctx = tvm.context("llvm", 0)
    # ex = tvm.relay.create_executor(mod=mod, ctx=ctx, target="llvm")

    # A = tvm.nd.array(np.array([[1, 2, 3], [4, 5, 6]], dtype="float32"), ctx=ctx)
    # B = tvm.nd.array(np.array([[8, 7, 6], [5, 4, 3]], dtype="float32"), ctx=ctx)
    # C = tvm.nd.array(np.array([[10, 11, 12], [13, 14, 15]], dtype="float32"), ctx=ctx)

    # result = ex.evaluate()(A, B, C)

    # print(result)

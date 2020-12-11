import tvm
import numpy as np
import tvm.ir
from tvm import relay, tir
from tvm.relay.dataflow_pattern import TupleGetItemPattern, is_op, wildcard
from tvm.relay.op.contrib.register import get_pattern_table, register_pattern_table
# from .register import register_pattern_table


@register_pattern_table("apu")
def pattern_table():
    """Get the APU compiler pattern table."""

    def multiply():
        pattern = is_op("multiply")(wildcard(), wildcard())
        return pattern

    def check_multiply(extract):
        """Check if multiply is supported."""
        return True

    return [
        ("apu_ops", multiply(), check_multiply)
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



#     """
#     a = wildcard()
#     b = wildcard()
#     c = wildcard()
#     add_node = is_op("add")(x, y)
#     return add_node

if __name__ == "__main__":
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

    mod = relay.transform.InferType()(mod)
    mod = relay.transform.VisualizeGraph("source.pdf")(mod)

    patterns = get_pattern_table("apu")
    mod = relay.transform.MergeComposite(patterns)(mod)
    mod = relay.transform.AnnotateTarget(targets=["llvm"])(mod)
    mod = relay.transform.InferType()(mod)
    mod = relay.transform.VisualizeGraph("post_MergeComposite.pdf")(mod)

    mod = relay.transform.MergeCompilerRegions()(mod)
    mod = relay.transform.InferType()(mod)
    mod = relay.transform.VisualizeGraph("post_MergeCompilerRegions.pdf")(mod)

    mod = relay.transform.PartitionGraph()(mod)
    mod = relay.transform.InferType()(mod)
    mod = relay.transform.VisualizeGraph("post_PartitionGraph.pdf")(mod)
    print(mod)

    ctx = tvm.context("llvm", 0)
    ex = tvm.relay.create_executor(mod=mod, ctx=ctx, target="llvm")

    A = tvm.nd.array(np.array([[1, 2, 3], [4, 5, 6]], dtype="float32"), ctx=ctx)
    B = tvm.nd.array(np.array([[8, 7, 6], [5, 4, 3]], dtype="float32"), ctx=ctx)
    C = tvm.nd.array(np.array([[10, 11, 12], [13, 14, 15]], dtype="float32"), ctx=ctx)

    print(A)

    result = ex.evaluate()(A, B, C)

    print(result)

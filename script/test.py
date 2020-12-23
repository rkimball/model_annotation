import tvm
import numpy as np
import tvm.ir
from tvm import relay, tir, autotvm
from tvm.relay.dataflow_pattern import TupleGetItemPattern, is_op, wildcard
from tvm.relay.op.contrib.register import get_pattern_table, register_pattern_table
from tvm.relay.op.annotation import compiler_begin, compiler_end
from tvm.relay.expr import Call, TupleGetItem, Var, Constant, Tuple
from tvm.ir import Op
from tvm.contrib import utils
import tvm.contrib.graph_runtime as runtime

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


def get_placement(expr):
    """ This method is called for each Call node in the graph. Return the targeted
    compiler for each Op or "default"
    """
    target_ops = ["multiply"]
    placement = "default"
    if isinstance(expr, Call):
        if isinstance(expr.op, Op):
            if expr.op.name in target_ops:
                placement = "vulkan"
    return placement


if __name__ == "__main__":
    mod = get_model()
    print(mod)

    # mod = relay.transform.AnnotateCompiler(get_placement)(mod)
    # mod = relay.transform.MergeCompilerRegions()(mod)
    # mod = relay.transform.PartitionGraph()(mod)
    # print(mod)

    # setup remote execution
    device = "4900hs"
    host = "tracker"
    port = 9191


    # target_host = "llvm -mtriple=x86_64-linux-win32"


    target = "vulkan"
    target_host = "llvm -mtriple=x86_64-linux-win32"

    remote = autotvm.measure.request_remote(device, host, port, timeout=1000)
    cpu_context = remote.cpu(0)
    vulkan_context = remote.vulkan(0)

    # target = {"llvm":"llvm -mcpu=znver2", "vulkan":"vulkan"}
    # target[cpu_context] = "llvm -mcpu=znver2" # windows llvm
    # target[vulkan_context] = "vulkan"

    with relay.build_config(opt_level=3):
        lib = relay.build(mod,
                        target=target,
                        target_host=target_host)

    temp = utils.tempdir()
    lib.export_library(temp.relpath("graphlib.tar"))
    # lib.get_source()
    remote.upload(temp.relpath("graphlib.tar"))
    rlib = remote.load_module("graphlib.tar")

    A = tvm.nd.array(np.array([[1, 2, 3], [4, 5, 6]], dtype="float32"), vulkan_context)
    B = tvm.nd.array(np.array([[8, 7, 6], [5, 4, 3]], dtype="float32"), vulkan_context)
    C = tvm.nd.array(np.array([[10, 11, 12], [13, 14, 15]], dtype="float32"), vulkan_context)

    module = runtime.GraphModule(rlib["default"](vulkan_context))
    # print(module.get_source())
    module.set_input(0, A)
    module.set_input(1, B)
    module.set_input(2, C)

    module.run()
    result = module.get_output(0)



    # ctx = tvm.context("llvm", 0)
    # ex = tvm.relay.create_executor(mod=mod, ctx=ctx, target="llvm")

    # result = ex.evaluate()(A, B, C)

    # [[1620. 1782. 1944.]
    #  [2106. 2268. 2430.]]
    print(result)

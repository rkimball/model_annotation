import argparse
import csv
import cv2
import logging
import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np
import onnx
import os
import pandas as pd
import re
import tensorflow as tf
try:
    tf_compat_v1 = tf.compat.v1
except ImportError:
    tf_compat_v1 = tf
import tf2onnx
import tflite
from PIL import Image

# tvm, relay, mxnet
import tvm
from tvm import relay
from tvm.relay import testing
import tvm.relay.testing.tf as tf_testing
from tvm.contrib import graph_runtime, utils, download
from tvm.contrib.debugger import debug_runtime
from tvm.contrib.download import download_testdata
from tvm import autotvm
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner


ONNX_TMP_FN = "model.onnx"

# Helper to extract from tgz file
def extract(path):
    import tarfile
    import zipfile
    dir_path = os.path.dirname(path)
    if path.endswith("tgz") or path.endswith("gz"):
        tar = tarfile.open(path)
        tar.extractall(path=dir_path)
        tar.close()
    elif path.endswith("zip"):
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(dir_path)
    else:
        raise RuntimeError('Could not decompress the file: ' + path)

# Helper to get default format
def get_default_format(model, via_onnx):
    if model.endswith("mxnet"):
        return "NCHW"
    elif model.endswith("onnx"):
        return "NCHW"
    elif model.endswith("tflite"):
        return "NHWC"
    elif model.endswith("tf") and not via_onnx:
        return "NHWC"
    elif model.endswith("tf") and via_onnx:
        return "NCHW"
    else:
        raise RuntimeError("Model framework of origin not recognized: {}".format(model))


def transform_image(image, transpose, mean, std, div_by):
    image = np.array(image) - np.array(mean)
    image /= np.array(std)
    image /= np.array(div_by)
    if transpose:
        image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image


def get_target_str(target):
    if "llvm" in str(target):
        return "llvm"
    elif "vulkan" in str(target):
        return "vulkan"
    elif "opencl" in str(target):
        return "opencl"
    else:
        assert False, "target not recognized"


def get_model(model, via_onnx, layout):
    if model.endswith("mxnet"):
        batch_size = 1
        input_shape = (batch_size, 3, 224, 224)
        dtype = "float32"
        if "mobilenet" in model:
            mod, params = testing.mobilenet.get_workload(batch_size=batch_size, dtype=dtype)
        elif "inception_v3" in model:
            input_shape = (batch_size, 3, 299, 299)
            mod, params = testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
        elif "resnet_18" in model:
            mod, params = testing.resnet.get_workload(
                num_layers=18, batch_size=batch_size, dtype=dtype
            )
        elif "resnet_50" in model:
            mod, params = testing.resnet.get_workload(
                num_layers=50, batch_size=batch_size, dtype=dtype
            )
        elif "vgg_16" in model:
            mod, params = testing.vgg.get_workload(
                num_layers=16, batch_size=batch_size, dtype=dtype
            )
        elif "vgg_19" in model:
            mod, params = testing.vgg.get_workload(
                num_layers=19, batch_size=batch_size, dtype=dtype
            )
        elif "densenet_121" in model:
            mod, params = testing.densenet.get_workload(
                densenet_size=121, batch_size=batch_size, dtype=dtype
            )
        elif "squeezenet_v1.0" in model:
            mod, params = testing.squeezenet.get_workload(
                batch_size=batch_size, version="1.0", dtype=dtype
            )
        elif "squeezenet_v1.1" in model:
            mod, params = testing.squeezenet.get_workload(
                batch_size=batch_size, version="1.1", dtype=dtype
            )
        shape_dict = {"data": input_shape}
    elif model == "amd_sinet_onnx":
        #
        # Description: AMD-provided background elimination model in onnx format.
        # Status: Imports and builds correctly, but build is slow.
        #
        iname = "input.1"
        dshape = (1, 3, 448, 448)
        shape_dict = {iname: dshape}
        # It expects the pb file saved locally to this file as SINet_448x448.onnx
        model_path = "SINet_448x448.onnx"
        onnx_model = onnx.load(model_path)
        # Import into Relay
        mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
    elif model == "amd_ssd_tf":
        #
        # Description: AMD-provided TF SSD-mobilenet model.
        # Status: Won't import either via ONNX or TF. Requires trimming of pre/post
        #         processing layers.
        #
        inames = ['image_tensor']
        onames = ['detection_boxes', 'detection_scores', 'detection_classes']
        N, C, H, W = 1, 3, 512, 512
        # It expects the pb file saved locally to this file as mobilenet_v2_ssd.pb
        # (originally dsp_apps_tasks_face_detection_model_frozen_inference_graph.pb)
        model_path = 'mobilenet_v2_ssd.pb'
        with tf_compat_v1.gfile.GFile(model_path, 'rb') as graphfile:
            graph_def = tf_compat_v1.GraphDef()
            graph_def.ParseFromString(graphfile.read())
            # Two ways to import: TF->Relay vs. TF->ONNX->Relay
            if via_onnx:
                onnx_fn = "model.onnx"
                ishape = (N, C, H, W)
                shape_dict = {inames[0] + ":0": ishape}
                graph = tf_compat_v1.import_graph_def(graph_def, name='')
                with tf_compat_v1.Session() as sess:
                    # Convert to ONNX
                    onnx_graph = tf2onnx.tfonnx.process_tf_graph(
                        sess.graph,
                        input_names=[i + ":0" for i in inames],
                        output_names=[i + ":0" for i in onames],
                        inputs_as_nchw=[i + ":0" for i in inames],
                        opset=11)
                    model_proto = onnx_graph.make_model("mobilenet_v2_ssd")
                    with open(onnx_fn, "wb") as f:
                        f.write(model_proto.SerializeToString())
                    # Import into Relay
                    onnx_model = onnx.load(onnx_fn)
                    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
            else:
                ishape = (N, H, W, C)
                shape_dict = {inames[0] + ":0": ishape}
                image_tensor, box, score, cls = tf_compat_v1.import_graph_def(
                        graph_def, name='', return_elements=[i + ":0" for i in inames+onames])
                graph_def = tf_testing.ProcessGraphDefParam(graph_def)
                with tf_compat_v1.Session() as sess:
                    shaped_graph_def = tf_testing.AddShapesToGraphDef(sess, onames)
                    # Import into Relay from graph def
                    mod, params = relay.frontend.from_tensorflow(shaped_graph_def,
                                                                 layout=None,
                                                                 outputs=onames,
                                                                 shape=shape_dict)
    elif model == "amd_mobilenetv2_tf":
        #
        # Description: AMD-provided TF mobilenet model.
        # Status: Imports and runs well. TF direct import works. Import via ONNX broken.
        #
        iname = 'input'
        oname = 'MobilenetV2/Predictions/Reshape_1'
        N, C, H, W = 1, 3, 224, 224
        # It expects the pb file saved locally to this file as mobilenet_v2.pb
        # (originally dsp_apps_tasks_scene_detection_xnnc_project_model_frozen_mobilenet_v2_100_224_sdc3_inf_graph2.pb)
        model_path = 'mobilenet_v2.pb'
        with tf_compat_v1.gfile.GFile(model_path, 'rb') as f:
            graph_def = tf_compat_v1.GraphDef()
            graph_def.ParseFromString(f.read())
            # Two ways to import: TF->Relay vs. TF->ONNX->Relay
            if via_onnx:
                ishape = (N, C, H, W)
                shape_dict = {iname + ":0": ishape}
                graph = tf_compat_v1.import_graph_def(graph_def, name='')
                with tf_compat_v1.Session() as sess:
                    # Convert to ONNX
                    onnx_graph = tf2onnx.tfonnx.process_tf_graph(
                        sess.graph,
                        input_names=[iname + ":0"],
                        output_names=[oname + ":0"],
                        inputs_as_nchw=[iname + ":0"])
                    model_proto = onnx_graph.make_model("mobilenet_v2")
                    with open(onnx_fn, "wb") as f:
                        f.write(model_proto.SerializeToString())
                    # Import into Relay
                    onnx_model = onnx.load(onnx_fn)
                    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
            else:
                ishape = (N, H, W, C)
                shape_dict = {iname + ":0": ishape}
                graph = tf.import_graph_def(graph_def, name='')
                # Call the utility to import the graph definition into default graph.
                graph_def = tf_testing.ProcessGraphDefParam(graph_def)
                # Add shapes to the graph.
                with tf_compat_v1.Session() as sess:
                    graph_def = tf_testing.AddShapesToGraphDef(sess, oname)
                    mod, params = relay.frontend.from_tensorflow(graph_def,
                                                                 layout=None,
                                                                 shape=shape_dict)
    elif model == "mobilenetv2_onnx":
        #
        # Description: off the shelf ONNX model from ONNX model zoo
        # Status: Imports and runs well.
        #
        # Perform download
        model_url = "https://github.com/onnx/models/raw/master/vision/classification/mobilenet/model/mobilenetv2-7.onnx"
        model_fn = os.path.basename(model_url)
        model_file = download_testdata(model_url, model_fn, module='onnx')
        onnx_model = onnx.load(model_file)
        # Extract shape_dict
        param_list = []
        shape_dict = {}
        for init_tensor in onnx_model.graph.initializer:
            param_list.append(init_tensor.name)
        for i in onnx_model.graph.input:
            i_name = i.name
            i_shape = i.type.tensor_type.shape.dim
            i_shape = [x.dim_value for x in i_shape]
            if i_name not in param_list:
                shape_dict[i_name] = i_shape
        # Import into Relay
        mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
    elif model == "mobilenetv2_tflite":
        #
        # Description: off the shelf TFLite model from TFLite model zoo
        # Status: Imports and runs well. Better perf achieved if transformed to NCHW.
        #
        # Perform download
        model_url = "https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224.tgz"
        model_fn = os.path.basename(model_url)
        model_path = download_testdata(model_url, model_fn, module="tf")
        model_dir = os.path.dirname(model_path)
        extract(model_path)
        # Load TFLite model
        tflite_model_file = os.path.join(model_dir, model_fn).replace("tgz", "tflite")
        if not os.path.exists(tflite_model_file):
            tflite_model_file = os.path.join(model_dir, model) + ".tflite"
        tflite_model_buf = open(tflite_model_file, "rb").read()
        tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
        # Get TFLite interpreter
        interpreter = tf.lite.Interpreter(model_path=tflite_model_file)
        # Derive input dictionaries
        shape_dict = {}
        type_dict = {}
        input_details = interpreter.get_input_details()
        for i in input_details:
            input_name = i["name"]
            shape_dict[input_name] = i["shape"].tolist()
            type_dict[input_name] = str(np.dtype(i["dtype"]))

        # Import from TFLite representation into Relay
        mod, params = relay.frontend.from_tflite(tflite_model,
                                                shape_dict=shape_dict,
                                                dtype_dict=type_dict)
    elif model == "ssd_onnx":
        #
        # Description: off the shelf ONNX model from ONNX Model Zoo
        # Status: Currently does not import correctly into Relay.
        #
        iname = "image_tensor:0"
        dshape = (1, 512, 512, 3)
        # Download model and load in mem
        url = "https://github.com/onnx/models/raw/master/vision/object_detection_segmentation/ssd-mobilenetv1/model/ssd_mobilenet_v1_10.onnx"
        model_file = download_testdata(url, 'ssd_mobilenet_v1_10.onnx', module='data')
        onnx_model = onnx.load(model_file)
        # Import into Relay
        mod, params = relay.frontend.from_onnx(onnx_model, {iname: dshape})
    else:
        raise RuntimeError("Model {} is not recognized".format(model))


    # Format change for ONNX from NCHW to NHWC
    if layout == "NHWC" and get_default_format(model, via_onnx) == "NCHW":
        desired_layouts = {
            'nn.conv2d': [layout, 'default'],
            'nn.depthwise_conv2d': [layout, 'default'],
        }
        seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
                                        relay.transform.ConvertLayout(desired_layouts),
                                        ])
        with tvm.transform.PassContext(opt_level=3):
            mod = seq(mod)
        # Update shape from NCHW to NHWC
        assert len(shape_dict) == 1
        for iname in shape_dict.keys():
            assert len(shape_dict[iname]) == 4
            shape_dict[iname] = [shape_dict[iname][i] for i in [0, 2, 3, 1]]
    # Format change for TF or TFLite to NCHW
    elif layout == "NCHW" and get_default_format(model, via_onnx) == "NHWC":
        desired_layouts = {
            'nn.conv2d': [layout, 'default'],
            'nn.depthwise_conv2d': [layout, 'default'],
        }
        seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
                                        relay.transform.ConvertLayout(desired_layouts)])
        with tvm.transform.PassContext(opt_level=3):
            mod = seq(mod)
        # Update shapes from NHWC to NCHW
        assert len(shape_dict) == 1
        for iname in shape_dict.keys():
            assert len(shape_dict[iname]) == 4
            assert len(shape_dict[iname]) == 4
            shape_dict[iname] = [shape_dict[iname][i] for i in [0, 3, 1, 2]]
    return mod, params, shape_dict

def tune_kernels(tasks,
                 measure_option,
                 tuner='gridsearch',
                 ntrial=2000,
                 early_stopping=None,
                 log_filename='tuning.log'):

    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, task in enumerate(tasks):
        prefix = "[Task %2d/%2d] " % (i+1, len(tasks))

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(task, loss_type='rank')
        elif tuner == 'ga':
            tuner_obj = GATuner(task, pop_size=50)
        elif tuner == 'random':
            tuner_obj = RandomTuner(task)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(task)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        # do tuning
        n_trial = min(len(task.config_space), ntrial)
        tuner_obj.tune(n_trial=n_trial,
                        early_stopping=early_stopping,
                        measure_option=measure_option,
                        callbacks=[
                                autotvm.callback.progress_bar(n_trial, prefix=prefix),
                                autotvm.callback.log_to_file(tmp_log_file)])

def tune_model(model,
               via_onnx,
               layout,
               target,
               target_host,
               rpc,
               ntrial,
               start,
               log_dir):

    print("tune_model")

    mod, params, _ = get_model(model, via_onnx, layout)

    tasks = autotvm.task.extract_from_program(mod["main"],
                                              target=target,
                                              target_host=target_host,
                                              params=params,
                                              ops=None)

    if rpc:
        runner=autotvm.RPCRunner(rpc["device"],
                rpc["host"],
                rpc["port"],
                timeout=10,
                number=20, repeat=3,
                min_repeat_ms=150,
                cooldown_interval=0.2,
                check_correctness=False)
    else:
        runner=autotvm.LocalRunner(
                timeout=10,
                number=20, repeat=3,
                min_repeat_ms=150,
                cooldown_interval=0.2,
                check_correctness=False)

    for idx, task in enumerate(tasks):
        if idx >= start:
            print("{} layer {}/{}".format(model, idx, len(tasks)))
            log_file = "{}_{}_{:02d}.log".format(model, get_target_str(target), idx)
            log_file = os.path.join(log_dir, log_file)
            tuning_option = {
                'log_filename': log_file,
                'tuner': 'ga',
                'ntrial': ntrial,
                'early_stopping': None,
                'measure_option': autotvm.measure_option(
                    builder=autotvm.LocalBuilder(timeout=30),
                    runner=runner
                ),
            }
            tune_kernels([task], **tuning_option)
            # pick best records to a cache file
            tmp_log_file = log_file + ".tmp"
            if os.path.exists(tmp_log_file):
                autotvm.record.pick_best(tmp_log_file, log_file)
                # os.remove(tmp_log_file)
    print("tune_model done")


def test_model(model,
               via_onnx,
               layout,
               target,
               target_host,
               rpc,
               apply_log,
               log_dir,
               debug,
               get_tar,
               im_check):
    print("test_model")

    mod, params, ishape = get_model(model, via_onnx, layout)

    # hetero prep
    mod = transform.AnnotateTarget("llvm")(mod)
    mod = transform.PartitionGraph()(mod)

    # This assumes there's only 1 input to the model
    inputs = list(ishape.keys())
    assert len(inputs) == 1
    iname = inputs[0]

    if rpc:
        remote = autotvm.measure.request_remote(rpc["device"],
                rpc["host"], rpc["port"], timeout=1000)
        if "llvm" in str(target):
            ctx = remote.cpu(0)
        elif "vulkan" in str(target):
            ctx = remote.vulkan(0)
        elif "opencl" in str(target):
            ctx = remote.cl(0)
    else:
        if "llvm" in str(target):
            ctx = tvm.cpu(0)
        elif "vulkan" in str(target):
            ctx = tvm.vulkan(0)
        elif "opencl" in str(target):
            ctx = tvm.cl(0)

    log_file = "{}_{}.log".format(model, get_target_str(target))
    log_file = os.path.join(log_dir, log_file)
    if not os.path.exists(log_file):
        reg_str = "{}_{}_[0-9]+\.log".format(model, get_target_str(target))
        regex = re.compile(reg_str)
        log_files = []
        for f in [x for x in os.listdir(log_dir) if os.path.isfile(os.path.join(log_dir, x))]:
            if regex.match(f):
                log_files.append(os.path.join(log_dir, f))
        log_files.sort()
        # Write all log files into one file
        with open(log_file, 'w') as outfile:
            for fname in log_files:
                with open(fname) as infile:
                    outfile.write(infile.read())

    if os.path.exists(log_file) and apply_log:
        print("Apply history best from %s" % log_file)
        with autotvm.apply_history_best(log_file):
            with relay.build_config(opt_level=3):
                graph, lib, params = relay.build(mod,
                                                 target=target,
                                                 target_host=target_host,
                                                 params=params)
            # Generate all in one tar
            if get_tar:
                with tvm.transform.PassContext(opt_level=3):
                    allinonelib = relay.build(mod,
                                              target=target,
                                              target_host=target_host,
                                              params=params)
                    allinonelib.export_library("graphlib.tar")
    else:
        print("Not applying history best")
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build(mod,
                                             target=target,
                                             target_host=target_host,
                                             params=params)
        if get_tar:
            with tvm.transform.PassContext(opt_level=3):
                allinonelib = relay.build(mod,
                                          target=target,
                                          target_host=target_host,
                                          params=params)
                allinonelib.export_library("graphlib.tar")

    if rpc:
        # Export library
        temp = utils.tempdir()
        lib.export_library(temp.relpath("graphlib.tar"))
        remote.upload(temp.relpath("graphlib.tar"))
        lib = remote.load_module("graphlib.tar")

    # Derive input dimensions
    if layout == "NCHW":
        input_dim = tuple(ishape[iname][2:4])
    elif layout == "NHWC":
        input_dim = tuple(ishape[iname][1:3])

    if model == "amd_sinet_onnx":
        img_path = "test.jpg"
    else:
        # Get an input image and labels
        img_url = 'https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true'
        img_path = download_testdata(img_url, 'cat.png', module='data')
        synset_url = ''.join(['https://gist.githubusercontent.com/zhreshold/',
                            '4d0b62f3d01426887599d4f7ede23ee5/raw/',
                            '596b27d23537e5a1b5751d2b0481ef172f58b539/',
                            'imagenet1000_clsid_to_human.txt'])
        synset_name = 'imagenet1000_clsid_to_human.txt'
        synset_path = download_testdata(synset_url, synset_name, module='data')
        with open(synset_path) as f:
            synset = eval(f.read())

    img = Image.open(img_path)
    img_orig = np.copy(img)
    image = img.resize(input_dim)
    # Transpose the image if the framework is NCHW
    is_NCHW = get_default_format(model, via_onnx) == "NCHW"
    mean = [123., 117., 104.]
    std = [58.395, 57.12, 57.375]
    div_by = 1
    if model == "amd_sinet_onnx":
        mean = [102.01, 110.30, 126.02]
        std = [62.61, 63.03, 66.99]
        div_by = 255.0
    x = transform_image(image, is_NCHW, mean, std, div_by)

    if debug:
        # build debug runtime
        m = debug_runtime.create(graph, lib, ctx)
    else:
        # build graph runtime
        m = graph_runtime.create(graph, lib, ctx)

    m.set_input(iname, tvm.nd.array(x.astype('float32')))
    m.set_input(**params)
    m.run()

    if debug:
        # Extract debug info
        eid = 0
        per_layer_performance = []
        debug_info = m.debug_datum
        total_time = sum(time_[0] for time_ in debug_info._time_list)
        for node, time_ in zip(debug_info._nodes_list, debug_info._time_list):
            for _ in range(debug_info.get_graph_node_output_num(node)):
                op = node["op"]
                if op != "param":
                    time_ms = round(time_[0] * 1000, 3)
                    time_percent = round((time_[0] / total_time) * 100, 3)
                    per_layer_performance.append(
                        {
                            "name": node["name"],
                            "op": op,
                            "time_ms": time_ms,
                            "time_percent": time_percent,
                            "shape": str(debug_info._output_tensor_list[eid].shape),
                        }
                    )
                eid += 1

        # Write stats to CSV file
        df = pd.DataFrame(per_layer_performance)
        df.to_csv("{}_{}_{}_{}_timing.csv".format(model, get_target_str(target), rpc["device"], "tuned" if apply_log else "untuned"), sep="\t")

    if im_check:
        if model == "amd_sinet_onnx":
            # Check correctness
            tvm_output = m.get_output(0).asnumpy()

            bg_name = "background.jpg"
            syn_bg = cv2.imread(bg_name)
            syn_bg = cv2.resize(syn_bg, (640, 480)) #(w,h)

            msk1 = np.squeeze(tvm_output) # [2, 448, 448]
            msk1 = msk1[1, :, :]
            msk1 = np.where(msk1 > 0.8, 1., 0.)
            msk1 = cv2.resize(msk1, (syn_bg.shape[1], syn_bg.shape[0]))
            msk1 = cv2.GaussianBlur(msk1, (7, 7), 3.0) #7.0

            seg_img = 0 * img_orig

            #modified thresholding method
            seg_img[:, :, 0] = img_orig[:, :, 0] * msk1 + syn_bg[:, :, 0] * (1 - msk1)
            seg_img[:, :, 1] = img_orig[:, :, 1] * msk1 + syn_bg[:, :, 1] * (1 - msk1)
            seg_img[:, :, 2] = img_orig[:, :, 2] * msk1 + syn_bg[:, :, 2] * (1 - msk1)

            seg_img = np.where(seg_img == 0, 0, seg_img)

            res = seg_img#cv2.hconcat([img_orig, seg_img])
            plt.imshow(res)
            plt.savefig('output.jpg')

        else:
            # Check correctness
            tvm_output = m.get_output(0)
            top_categories = np.argsort(tvm_output.asnumpy()[0])

            # Report top-5 classification results
            print("\n{} prediction".format(model))
            print("\t#1:", synset[top_categories[-1]])
            print("\t#2:", synset[top_categories[-2]])
            print("\t#3:", synset[top_categories[-3]])
            print("\t#4:", synset[top_categories[-4]])
            print("\t#5:", synset[top_categories[-5]])

    # Evaluate time
    ftimer = m.module.time_evaluator("run", ctx, number=10, repeat=10)
    prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
    print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
            (np.mean(prof_res), np.std(prof_res)))
    print("test_model done")

    return [model, "{}".format(input_dim), auto, np.mean(prof_res), np.std(prof_res)]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Tuning parameters
    parser.add_argument("--tune", action="store_true",
                        help="If set, run tuning on the model.")
    parser.add_argument("--ntrial", type=int, default=2000,
                        help="Number of autoTVM trials (used when passing --tune).")
    parser.add_argument("--start-layer", type=int, default=0,
                        help="Start tuning from layer N (default 0).")
    parser.add_argument("--log-dir", type=str, default="../logs_v1000/",
                        help="Dir of autoTVM logs.")
    # High level workflow params
    parser.add_argument("--apply-log", action="store_true",
                        help="If set, apply autotuning log.")
    parser.add_argument("--check-on-image", action="store_true",
                        help="If set, runs the network against a cat image to check output (only works for classifiation workloads).")
    parser.add_argument("--debug", action="store_true",
                        help="If set, obtains per-layer breakdown of execution time.")
    parser.add_argument("--get-tar", action="store_true",
                        help="If set, generates locally a tarball.")
    # Workload args
    parser.add_argument("--network", type=str, default="amd_sinet_onnx",
                        help="Model to evaluate.")
    parser.add_argument("--layout", type=str, choices=["NCHW", "NHWC"], default="NCHW",
                        help="Layout to transpose operators to.")
    parser.add_argument("--via-onnx", action="store_true", help="If set, import via ONNX")
    # TVM target args
    parser.add_argument("--target", type=str, choices=["llvm -mcpu=znver1",
                                                       "llvm -mcpu=znver2",
                                                       "vulkan",
                                                       "llvm"],
                        default="llvm -mcpu=znver1", help="TVM target string.")
    parser.add_argument("--target-os", type=str, choices=["windows", "linux"],
                        default="linux", help="TVM target operating system.")
    # RPC tracker args
    parser.add_argument("--remote", action="store_true",
                        help="Program target over RPC (requires valid tracker params).")
    parser.add_argument("--rpc-host", type=str, default='tracker',
                        help="RPC tracker host name.")
    parser.add_argument("--rpc-port", type=int, default=9191,
                        help="RPC tracker port.")
    parser.add_argument("--rpc-device", type=str, default="v1000", choices=["4900hs", "v1000"],
                        help="Device string used to register target to the tracker.")
    args = parser.parse_args()
    logging.basicConfig()
    print("Start " + args.network + " on " + args.rpc_device + " target " + args.target)

    # Tracker dict, set to none if local
    tracker_info = None if args.remote==False else {"host": args.rpc_host,
                                                    "port": args.rpc_port,
                                                    "device": args.rpc_device}

    # Derive target host string
    target_host = None
    if args.target == "vulkan":
        if args.target_os == "windows":
            target_host = "llvm -mtriple=x86_64-linux-win32"
        elif args.target_os == "linux":
            target_host = "llvm -mtriple=x86_64-linux-gnu"

    # Change LLVM target string if we target windows
    target = args.target
    if args.target_os == "windows" and "llvm" in args.target:
        target += " -mtriple=x86_64-pc-win32"

    # Tune
    if args.tune:
        tune_model(
            model=args.network,
            via_onnx=args.via_onnx,
            layout=args.layout,
            target=target,
            target_host=target_host,
            rpc=tracker_info,
            ntrial=args.ntrial,
            start=args.start_layer,
            log_dir=args.log_dir)

    # Experimental for inference
    res = [["model", "(H, W)", "auto", "ms(mean)", "ms(std)"]]
    for auto in [False]:
            res.append(test_model(
                model=args.network,
                via_onnx=args.via_onnx,
                layout=args.layout,
                target=target,
                target_host=target_host,
                rpc=tracker_info,
                apply_log=args.apply_log,
                log_dir=args.log_dir,
                debug=args.debug,
                get_tar=args.get_tar,
                im_check=args.check_on_image))

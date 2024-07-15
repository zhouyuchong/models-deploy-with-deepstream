'''
Author: zhouyuchong
Date: 2024-07-04 15:45:12
Description: 
LastEditors: zhouyuchong
LastEditTime: 2024-07-15 16:31:25
'''
import os
import sys
import argparse
import time
import torch
import glob
import cv2
import tqdm

from PIL import Image
from cuda import cuda, cudart

import numpy as np
import tensorrt as trt

from utils.evaluation import *
from utils.datasets import *
from utils.common import *

MAX_DET = 5

TRT_LOGGER = trt.Logger()

class TensorRTInfer:
    """
    Implements inference for TensorRT engine.
    """

    def __init__(self, engine_path, mode='max'):
        """
        :param engine_path: The path to the serialized engine to load from disk.
        :param max: infer mode, 'max' or 'min', batch size
        """
        # Load TRT engine
        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            assert runtime
            self.engine = runtime.deserialize_cuda_engine(f.read())
        assert self.engine
        self.context = self.engine.create_execution_context()
        assert self.context

        # Setup I/O bindings
        self.inputs = []
        self.outputs = []
        self.allocations = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            is_input = False
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                is_input = True
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            shape = self.context.get_tensor_shape(name)
            if is_input and shape[0] < 0:
                assert self.engine.num_optimization_profiles > 0
                profile_shape = self.engine.get_tensor_profile_shape(name, 0)
                assert len(profile_shape) == 3  # min,opt,max
                # Set the *max* profile as binding shape
                if mode == 'max':
                    self.context.set_input_shape(name, profile_shape[2])
                elif mode == 'min':
                    self.context.set_input_shape(name, profile_shape[0])
                shape = self.context.get_tensor_shape(name)
            if is_input:
                self.batch_size = shape[0]
            size = dtype.itemsize
            for s in shape:
                size *= s
            allocation = cuda_call(cudart.cudaMalloc(size))
            host_allocation = None if is_input else np.zeros(shape, dtype)
            binding = {
                "index": i,
                "name": name,
                "dtype": dtype,
                "shape": list(shape),
                "allocation": allocation,
                "host_allocation": host_allocation,
            }
            self.allocations.append(allocation)
            if is_input:
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)
            print(
                "{} '{}' with shape {} and dtype {}".format(
                    "Input" if is_input else "Output",
                    binding["name"],
                    binding["shape"],
                    binding["dtype"],
                )
            )

        assert self.batch_size > 0
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.allocations) > 0

    def input_spec(self):
        """
        Get the specs for the input tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the input tensor and its (numpy) datatype.
        """
        return self.inputs[0]["shape"], self.inputs[0]["dtype"]

    def output_spec(self):
        """
        Get the specs for the output tensors of the network. Useful to prepare memory allocations.
        :return: A list with two items per element, the shape and (numpy) datatype of each output tensor.
        """
        specs = []
        for o in self.outputs:
            specs.append((o["shape"], o["dtype"]))
        return specs

    def infer(self, batch):
        """
        Execute inference on a batch of images.
        :param batch: A numpy array holding the image batch.
        :return A list of outputs as numpy arrays.
        """
        # Copy I/O and Execute
        memcpy_host_to_device(self.inputs[0]["allocation"], batch)
        self.context.execute_v2(self.allocations)
        for o in range(len(self.outputs)):
            memcpy_device_to_host(
                self.outputs[o]["host_allocation"], self.outputs[o]["allocation"]
            )
        return [o["host_allocation"] for o in self.outputs]


def main(args):
    if args.input:
        if not is_file_or_folder(args.input):
            print("path not exists")
            sys.exit(1)
        elif is_file_or_folder(args.input) == 1:
            print("\nSingle image inference\n")
            trt_infer = TensorRTInfer(engine_path=args.engine, mode='min')
            spec = trt_infer.input_spec()
            input_data, ori_image = preprocess(args.input, spec[0])
            preds = infer_single(model=trt_infer, image=input_data, netshape=spec[0], imgshape=ori_image.shape, conf=args.conf, )
            draw_results(preds, ori_image, args.input.replace('.jpg', '_output.jpg'))
        elif is_file_or_folder(args.input) == 2 and args.val:
            print("\nValidation mode\n")
            trt_infer = TensorRTInfer(engine_path=args.engine, mode='min')
            spec = trt_infer.input_spec()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            evaluation = CocoDetectionEvaluator('/opt/nvidia/deepstream/deepstream-6.3/sources/model/yolov10/export/coco80.names', device)
            # 数据集加载
            val_dataset = TensorDataset('/opt/nvidia/deepstream/deepstream-6.3/sources/datasets/coco2017/coco2017/val.txt', spec[0][2], spec[0][3], False)
            mAP05 = evaluation.compute_map(val_dataset, trt_infer, (spec[0][2], spec[0][3]))
            print(mAP05)
    else:
        print("No input provided, running in benchmark mode")
        trt_infer = TensorRTInfer(engine_path=args.engine)
        spec = trt_infer.input_spec()
        batch = 255 * np.random.rand(*spec[0]).astype(spec[1])
        iterations = 200
        times = []
        for i in range(20):  # GPU warmup iterations
            trt_infer.infer(batch)
        for i in range(iterations):
            start = time.time()
            trt_infer.infer(batch)
            times.append(time.time() - start)
            print("Iteration {} / {}".format(i + 1, iterations), end="\r")
        print("Benchmark results include time for H2D and D2H memory copies")
        print("Average Latency: {:.3f} ms".format(1000 * np.average(times)))
        print(
            "Average Throughput: {:.1f} ips".format(
                trt_infer.batch_size / np.average(times)
            )
        )

    print("Finished Processing")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--engine",
        default=None,
        required=True,
        help="The serialized TensorRT engine",
    )
    parser.add_argument(
        "-i", "--input", default=None, help="Path to the image or directory to process"
    )
    parser.add_argument(
        "-v",
        "--val",
        action="store_true",
        help="Directory where to save the visualization results",
    )
    parser.add_argument(
        "--conf",
        default=0.25,
        type=float,
        help="The confidence threshold for the model",
    )

    args = parser.parse_args()
    for key,value in vars(args).items():
        print(key, ":", value)
    main(args)
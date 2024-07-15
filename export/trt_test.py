'''
Author: zhouyuchong
Date: 2024-07-04 15:45:12
Description: 
LastEditors: zhouyuchong
LastEditTime: 2024-07-15 16:09:49
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

from loguru import logger
# from evaluation import *
# from datasets import *

from utils import *

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
            logger.debug(f"Binding {name}, dtype {dtype}, shape {self.context.get_tensor_shape(name)}")
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

    def process(self, batch, scales=None, nms_threshold=None):
        """
        Execute inference on a batch of images. The images should already be batched and preprocessed, as prepared by
        the ImageBatcher class. Memory copying to and from the GPU device will be performed here.
        :param batch: A numpy array holding the image batch.
        :param scales: The image resize scales for each image in this batch. Default: No scale postprocessing applied.
        :return: A nested list for each image in the batch and each detection in the list.
        """
        # Run inference
        outputs = self.infer(batch)

        # Process the results
        nums = outputs[0]
        boxes = outputs[1]
        scores = outputs[2]
        classes = outputs[3]
        detections = []
        normalized = np.max(boxes) < 2.0
        for i in range(self.batch_size):
            detections.append([])
            for n in range(int(nums[i])):
                scale = self.inputs[0]["shape"][2] if normalized else 1.0
                if scales and i < len(scales):
                    scale /= scales[i]
                if nms_threshold and scores[i][n] < nms_threshold:
                    continue
                detections[i].append(
                    {
                        "ymin": boxes[i][n][0] * scale,
                        "xmin": boxes[i][n][1] * scale,
                        "ymax": boxes[i][n][2] * scale,
                        "xmax": boxes[i][n][3] * scale,
                        "score": scores[i][n],
                        "class": int(classes[i][n]),
                    }
                )
        return detections

def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
    dw = x[..., 2] / 2  # half-width
    dh = x[..., 3] / 2  # half-height
    y[..., 0] = x[..., 0] - dw  # top left x
    y[..., 1] = x[..., 1] - dh  # top left y
    y[..., 2] = x[..., 0] + dw  # bottom right x
    y[..., 3] = x[..., 1] + dh  # bottom right y
    return y

def v10postprocess(preds, max_det, nc=80):
    assert(4 + nc == preds.shape[-1])
    boxes, scores = preds.split([4, nc], dim=-1)
    max_scores = scores.amax(dim=-1)
    print(max_scores.shape)
    max_scores, index = torch.topk(max_scores, max_det, dim=-1)
    index = index.unsqueeze(-1)
    boxes = torch.gather(boxes, dim=1, index=index.repeat(1, 1, boxes.shape[-1]))
    scores = torch.gather(scores, dim=1, index=index.repeat(1, 1, scores.shape[-1]))

    scores, index = torch.topk(scores.flatten(1), max_det, dim=-1)
    labels = index % nc
    index = index // nc
    boxes = boxes.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, boxes.shape[-1]))
    return boxes, scores, labels

def infer_single(model, image):
    preds = model.infer(image)[0]
    mask = preds[..., 4] > 0.1
    preds = [p[mask[idx]] for idx, p in enumerate(preds)]
    return preds

def draw_results(preds, image_path):
    image = cv2.imread(image_path)
    ori_shape = image.shape
    scale = 640/480
    print(ori_shape)
    for batch in preds:
        re = batch
        for i in re:
            print(i)
        
            x1, y1, x2, y2 = i[:4]
            cv2.rectangle(image, (int(x1), int(y1/scale)), (int(x2), int(y2/scale)), (0, 255, 0), 2)

    cv2.imwrite("output.jpg", image)

def main(args):
    if args.input:
        if not is_file_or_folder(args.input):
            logger.error("path not exists")
            sys.exit(1)
        elif is_file_or_folder(args.input) == 1:
            trt_infer = TensorRTInfer(engine_path=args.engine, mode='min')
            logger.info("Single image inference")
            spec = trt_infer.input_spec()
            logger.debug("input shape: {}".format(spec[0]))

            img = cv2.imread(args.input)

            # Get the height and width of the input image
            img_height, img_width = img.shape[:2]

            # Convert the image color space from BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Resize the image to match the input shape
            img = cv2.resize(img, (spec[0][2], spec[0][3]))

            # Normalize the image data by dividing it by 255.0
            image_data = np.array(img) / 255.0

            # Transpose the image to have the channel dimension as the first dimension
            image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

            # Expand the dimensions of the image data to match the expected input shape
            image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

            # Return the preprocessed image data

            # # image = cv2.imread(args.input)
            # # image = image.astype(np.float32)
            # # logger.debug(image.shape)
            # # image /= 255.0
            # # image = np.transpose(image, [2, 0, 1])

            image = np.array(image_data, dtype=np.float32, order="C")
            preds = infer_single(model=trt_infer, image=image)
            print(preds)
            draw_results(preds, args.input)
        elif is_file_or_folder(args.input) == 2 and args.val:
            logger.info("valuation mode")
            trt_infer = TensorRTInfer(engine_path=args.engine, mode='min')
            spec = trt_infer.input_spec()
            print(spec)
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

    args = parser.parse_args()
    logger.debug(vars(args))
    main(args)
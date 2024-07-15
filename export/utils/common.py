#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import ctypes
import numpy as np
import tensorrt as trt
import torch
import time
import cv2
import loguru

from typing import Optional, List, Union
from cuda import cuda, cudart


def check_cuda_err(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError("Cuda Error: {}".format(err))
    if isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError("Cuda Runtime Error: {}".format(err))
    else:
        raise RuntimeError("Unknown error type: {}".format(err))

def cuda_call(call):
    err, res = call[0], call[1:]
    # loguru.logger.debug("{} | {}".format(err, res))
    check_cuda_err(err)
    if len(res) == 1:
        res = res[0]
    return res


class HostDeviceMem:
    """Pair of host and device memory, where the host memory is wrapped in a numpy array"""
    def __init__(self, size: int, dtype: Optional[np.dtype] = None):
        dtype = dtype or np.dtype(np.uint8)
        nbytes = size * dtype.itemsize
        host_mem = cuda_call(cudart.cudaMallocHost(nbytes))
        pointer_type = ctypes.POINTER(np.ctypeslib.as_ctypes_type(dtype))

        self._host = np.ctypeslib.as_array(ctypes.cast(host_mem, pointer_type), (size,))
        self._device = cuda_call(cudart.cudaMalloc(nbytes))
        self._nbytes = nbytes

    @property
    def host(self) -> np.ndarray:
        return self._host

    @host.setter
    def host(self, data: Union[np.ndarray, bytes]):
        if isinstance(data, np.ndarray):
            if data.size > self.host.size:
                raise ValueError(
                    f"Tried to fit an array of size {data.size} into host memory of size {self.host.size}"
                )
            np.copyto(self.host[:data.size], data.flat, casting='safe')
        else:
            assert self.host.dtype == np.uint8
            self.host[:self.nbytes] = np.frombuffer(data, dtype=np.uint8)

    @property
    def device(self) -> int:
        return self._device

    @property
    def nbytes(self) -> int:
        return self._nbytes

    def __str__(self):
        return f"Host:\n{self.host}\nDevice:\n{self.device}\nSize:\n{self.nbytes}\n"

    def __repr__(self):
        return self.__str__()

    def free(self):
        cuda_call(cudart.cudaFree(self.device))
        cuda_call(cudart.cudaFreeHost(self.host.ctypes.data))


# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
# If engine uses dynamic shapes, specify a profile to find the maximum input & output size.
def allocate_buffers(engine: trt.ICudaEngine, profile_idx: Optional[int] = None):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda_call(cudart.cudaStreamCreate())
    tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    for binding in tensor_names:
        # get_tensor_profile_shape returns (min_shape, optimal_shape, max_shape)
        # Pick out the max shape to allocate enough memory for the binding.
        shape = engine.get_tensor_shape(binding) if profile_idx is None else engine.get_tensor_profile_shape(binding, profile_idx)[-1]
        shape_valid = np.all([s >= 0 for s in shape])
        if not shape_valid and profile_idx is None:
            raise ValueError(f"Binding {binding} has dynamic shape, " +\
                "but no profile was specified.")
        size = trt.volume(shape)
        trt_type = engine.get_tensor_dtype(binding)

        # Allocate host and device buffers
        if trt.nptype(trt_type):
            dtype = np.dtype(trt.nptype(trt_type))
            bindingMemory = HostDeviceMem(size, dtype)
        else: # no numpy support: create a byte array instead (BF16, FP8, INT4)
            size = int(size * trt_type.itemsize)
            bindingMemory = HostDeviceMem(size)

        # Append the device buffer to device bindings.
        bindings.append(int(bindingMemory.device))

        # Append to the appropriate list.
        if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
            inputs.append(bindingMemory)
        else:
            outputs.append(bindingMemory)
    return inputs, outputs, bindings, stream


# Frees the resources allocated in allocate_buffers
def free_buffers(inputs: List[HostDeviceMem], outputs: List[HostDeviceMem], stream: cudart.cudaStream_t):
    for mem in inputs + outputs:
        mem.free()
    cuda_call(cudart.cudaStreamDestroy(stream))


# Wrapper for cudaMemcpy which infers copy size and does error checking
def memcpy_host_to_device(device_ptr: int, host_arr: np.ndarray):
    nbytes = host_arr.size * host_arr.itemsize
    # loguru.logger.debug("Copying {} bytes from host to device".format(nbytes))
    cuda_call(cudart.cudaMemcpy(device_ptr, host_arr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))
    # loguru.logger.debug("ret: {}".format(ret))

# Wrapper for cudaMemcpy which infers copy size and does error checking
def memcpy_device_to_host(host_arr: np.ndarray, device_ptr: int):
    nbytes = host_arr.size * host_arr.itemsize
    cuda_call(cudart.cudaMemcpy(host_arr, device_ptr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))


def _do_inference_base(inputs, outputs, stream, execute_async_func):
    # Transfer input data to the GPU.
    kind = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
    [cuda_call(cudart.cudaMemcpyAsync(inp.device, inp.host, inp.nbytes, kind, stream)) for inp in inputs]
    # Run inference.
    execute_async_func()
    # Transfer predictions back from the GPU.
    kind = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
    [cuda_call(cudart.cudaMemcpyAsync(out.host, out.device, out.nbytes, kind, stream)) for out in outputs]
    # Synchronize the stream
    cuda_call(cudart.cudaStreamSynchronize(stream))
    # Return only the host outputs.
    return [out.host for out in outputs]


# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, engine, bindings, inputs, outputs, stream):
    def execute_async_func():
        context.execute_async_v3(stream_handle=stream)
    # Setup context tensor address.
    num_io = engine.num_io_tensors
    for i in range(num_io):
        context.set_tensor_address(engine.get_tensor_name(i), bindings[i])
    return _do_inference_base(inputs, outputs, stream, execute_async_func)


# Sets up the builder to use the timing cache file, and creates it if it does not already exist
def setup_timing_cache(config: trt.IBuilderConfig, timing_cache_path: os.PathLike):
    buffer = b""
    if os.path.exists(timing_cache_path):
        with open(timing_cache_path, mode="rb") as timing_cache_file:
            buffer = timing_cache_file.read()
    timing_cache: trt.ITimingCache = config.create_timing_cache(buffer)
    config.set_timing_cache(timing_cache, True)


# Saves the config's timing cache to file
def save_timing_cache(config: trt.IBuilderConfig, timing_cache_path: os.PathLike):
    timing_cache: trt.ITimingCache = config.get_timing_cache()
    with open(timing_cache_path, "wb") as timing_cache_file:
        timing_cache_file.write(memoryview(timing_cache.serialize()))

def is_file_or_folder(path):
    if os.path.isfile(path):
        return 1
    elif os.path.isdir(path):
        return 2
    else:
        return 0
    
def parse_yolo_format(path):
    gt_li = []
    with open(path, 'r') as f:
        for line in f.readlines():
            gt = line.strip().split(' ')
            gt = [float(x) for x in gt]
            # gt = np.array(gt)
            gt_li.append(gt)
    gt_combined = np.array(gt_li)
    return gt_combined
        # return [line.strip() for line in f.readlines()]

def match_predictions(pred_classes, true_classes, iou, use_scipy=False):
    """
    Matches predictions to ground truth objects (pred_classes, true_classes) using IoU.

    Args:
        pred_classes (torch.Tensor): Predicted class indices of shape(N,).
        true_classes (torch.Tensor): Target class indices of shape(M,).
        iou (torch.Tensor): An NxM tensor containing the pairwise IoU values for predictions and ground of truth
        use_scipy (bool): Whether to use scipy for matching (more precise).

    Returns:
        (torch.Tensor): Correct tensor of shape(N,10) for 10 IoU thresholds.
    """
    iouv = torch.linspace(0.5, 0.95, 10)  
    # Dx10 matrix, where D - detections, 10 - IoU thresholds
    correct = np.zeros((pred_classes.shape[0], iouv.shape[0])).astype(bool)
    # LxD matrix where L - labels (rows), D - detections (columns)
    correct_class = true_classes[:, None] == pred_classes
    iou = iou * correct_class  # zero out the wrong classes
    iou = iou.cpu().numpy()
    for i, threshold in enumerate(iouv.cpu().tolist()):
        matches = np.nonzero(iou >= threshold)  # IoU > threshold and classes match
        matches = np.array(matches).T
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)

def box_iou(box1, box2, eps=1e-7):
    """
    Calculate intersection-over-union (IoU) of boxes. Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Based on https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py

    Args:
        box1 (torch.Tensor): A tensor of shape (N, 4) representing N bounding boxes.
        box2 (torch.Tensor): A tensor of shape (M, 4) representing M bounding boxes.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): An NxM tensor containing the pairwise IoU values for every element in box1 and box2.
    """

    # NOTE: need float32 to get accurate iou values
    box1 = torch.as_tensor(box1, dtype=torch.float32)
    box2 = torch.as_tensor(box2, dtype=torch.float32)
    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

def nms(boxes, scores, overlap_threshold):
    """
    boxes: [N, 4]，x1, y1, x2, y2
    scores: [N,]，每个边界框的置信度
    overlap_threshold: IOU阈值，用于判断是否重叠
    """
    # 获取边界框的数量
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    # 计算每个边界框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    # 对置信度进行排序，并获取排序后的索引
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        # 获取当前置信度最高的边界框的索引
        i = order[0]
        keep.append(i)
        
        # 计算当前边界框与剩下边界框的交集面积
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        
        # 计算IOU
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        
        # 保留IOU小于阈值的边界框
        inds = np.where(ovr <= overlap_threshold)[0]
        order = order[inds + 1]
    
    return keep


def infer_single(model, image, netshape, imgshape, conf=0.25):
    start_time = time.time()
    preds = model.infer(image)[0]
    end_time = time.time()
    print(f"\nTime taken for inference: {end_time - start_time} seconds\n")
    mask = preds[..., 4] > conf
    
    preds = [p[mask[idx]] for idx, p in enumerate(preds)]
    preds = postprocess(preds, netshape, imgshape)
    return preds

def postprocess(preds, net_shape, img_shape):
    img_height, img_width = img_shape[:2]
    x_scale = img_width / net_shape[2]
    y_scale = img_height / net_shape[3] 
    p = []
    for b in preds:
        b[:, [0, 2]] *= x_scale
        b[:, [1, 3]] *= y_scale
        keep = nms(b[:, :4], b[:, 4], 0.5)
        preds_afternms = b[keep]
        p.append(preds_afternms)
    return p

def preprocess(input_path, netshape):
    img = cv2.imread(input_path)
    input_data = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_data = cv2.resize(input_data, (netshape[2], netshape[3]))
    image_data = np.array(input_data) / 255.0
    image_data = np.transpose(image_data, (2, 0, 1))  # Channel first
    image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
    image = np.array(image_data, dtype=np.float32, order="C")
    return image, img

def draw_results(preds, image, output_path):
    for batch in preds:
        re = batch
        for i in re:        
            x1, y1, x2, y2 = i[:4]
            conf = round(i[4], 2)
            label = int(i[5])
            color = tuple(np.random.randint(256, size=3))
            color = (int(color[0]), int(color[1]), int(color[2]))
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(image, str(label), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(image, str(conf), (int(x2 - 40), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.imwrite(output_path, image)
    print(f"\noutput saved to {output_path}\n")
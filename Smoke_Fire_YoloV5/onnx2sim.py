import sys
import os
sys.path.append(os.getcwd())
from PIL import Image, ImageDraw, ImageFont
#from model.STN import STNet
import numpy as np
import argparse
import torch
import time
import cv2
import onnx

f = './weights/smoke.onnx'
model_onnx = onnx.load(f)  # load onnx model
dynamic = False
# onnx.checker.check_model(model_onnx)  # check onnx model

# Simplify

import onnxsim

model_onnx, check = onnxsim.simplify(
    model_onnx,
    dynamic_input_shape=dynamic,
    input_shapes = None)
assert check, 'assert check failed'
onnx.save(model_onnx, f)


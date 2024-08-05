'''
Author: zhouyuchong
Date: 2024-05-30 11:15:58
Description: 
LastEditors: zhouyuchong
LastEditTime: 2024-07-09 17:14:02
'''
import os
import argparse
from loguru import logger
import tensorrt as trt
import math

TRT_LOGGER = trt.Logger()

def build_engine(onnx_file_path, inputname, engine_file_path="", set_input_shape=None):
    """Takes an ONNX file and creates a TensorRT engine to run inference with"""
    network_creation_flag = 0
    network_creation_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
        network_creation_flag
    ) as network, builder.create_builder_config() as config, trt.OnnxParser(
        network, TRT_LOGGER
    ) as parser, trt.Runtime(
        TRT_LOGGER
    ) as runtime:
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, 1 << 28
        )  # 256MiB
        # Parse model file
        if not os.path.exists(onnx_file_path):
            print(
                "ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.".format(
                    onnx_file_path
                )
            )
            exit(0)
        print("Loading ONNX file from path {}...".format(onnx_file_path))
        with open(onnx_file_path, "rb") as model:
            print("Beginning ONNX file parsing")
            if not parser.parse(model.read()):
                print("ERROR: Failed to parse the ONNX file.")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        # set input shape
        profile = builder.create_optimization_profile()
        logger.debug("total input layer: {}".format(network.num_inputs))
        logger.debug(network.num_outputs)
        output = network.get_output(0)
        logger.debug(output.shape)
        for i in range(network.num_inputs):
            input = network.get_input(i)
        #     assert input.shape[0] == -1
            logger.debug("input layer-{}: {}".format(i, input.name))
        profile.set_shape(inputname, set_input_shape[0], set_input_shape[1], set_input_shape[2])
        config.add_optimization_profile(profile)
        logger.debug("build, may take a while...")

        plan = builder.build_serialized_network(network, config)
        engine = runtime.deserialize_cuda_engine(plan)
        print("Completed creating Engine")
        with open(engine_file_path, "wb") as f:
            f.write(plan)
        return engine
    
    
    
def main(args):
    trt_file_name = args.onnx.replace('.onnx', '_bs{}.trt'.format(args.batch))
    input_shape = [(1, 3, args.size, args.size), (math.ceil(args.batch/2), 3, args.size, args.size), (args.batch, 3, args.size, args.size)]
    logger.debug("set input shape: {}".format(input_shape))
    build_engine(args.onnx, args.name, trt_file_name, input_shape)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', type=str, default='', help='onnx path', required=True)
    parser.add_argument('-s', '--size', type=int, default=640, help='input shape')
    parser.add_argument('-b', '--batch', type=int, default=1, help='max batch size')
    parser.add_argument('-n', '--name', type=str, default='images', help='model input layer name')
    args = parser.parse_args()
    main(args=args)

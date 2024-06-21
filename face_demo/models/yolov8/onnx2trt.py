'''
Author: zhouyuchong
Date: 2024-03-27 14:14:55
Description: 
LastEditors: zhouyuchong
LastEditTime: 2024-05-27 11:22:08
'''
import tensorrt as trt

if __name__ == "__main__":
    logger = trt.Logger(trt.Logger.WARNING)

    builder = trt.Builder(logger) 
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    # network = builder.create_network(1 << 2)
    parser = trt.OnnxParser(network, logger)

    success = parser.parse_from_file("yolov8s-pose.onnx")
    if not success:
        print("parse error occured!")
        exit
    # builder.max_workspace_size = 1<<28
    builder.max_batch_size = 64

    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30
    profile = builder.create_optimization_profile()
    profile.set_shape("input", (1,3,640,640), (1,3,640,640), (1,3,640,640))
    config.add_optimization_profile(profile)
    
    serialized_engine = builder.build_serialized_network(network, config)
    with open("yolov8s-pose.engine", "wb") as f:
        f.write(serialized_engine)

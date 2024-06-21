'''
Author: zhouyuchong
Date: 2024-05-30 11:15:58
Description: 
LastEditors: zhouyuchong
LastEditTime: 2024-05-30 11:16:00
'''
import tensorrt as trt

if __name__ == "__main__":
    logger = trt.Logger(trt.Logger.WARNING)

    builder = trt.Builder(logger) 
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    # network = builder.create_network(1 << 2)
    parser = trt.OnnxParser(network, logger)

    success = parser.parse_from_file("RepVGG_8emotions_bs1.onnx")
    if not success:
        print("parse error occured!")
        exit
    # builder.max_workspace_size = 1<<28
    builder.max_batch_size = 64

    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30
    profile = builder.create_optimization_profile()
    profile.set_shape("input", (1,3,224,224), (1,3,224,224), (1,3,224,224))
    config.add_optimization_profile(profile)
    
    serialized_engine = builder.build_serialized_network(network, config)
    with open("RepVGG_8emotions_bs1.engine", "wb") as f:
        f.write(serialized_engine)

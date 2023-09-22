import tensorrt as trt

if __name__ == "__main__":
    logger = trt.Logger(trt.Logger.WARNING)

    builder = trt.Builder(logger) 
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    # network = builder.create_network(1 << 2)
    parser = trt.OnnxParser(network, logger)

    success = parser.parse_from_file("./weights/smoke.onnx")
    if not success:
        print("parse error occured!")
        exit

    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30
    profile = builder.create_optimization_profile()
    profile.set_shape("images", (1,3,640,640), (1,3,640,640), (1,3,640,640))
    config.add_optimization_profile(profile)
    
    serialized_engine = builder.build_serialized_network(network, config)
    with open("./weights/smoke.engine", "wb") as f:
        f.write(serialized_engine)


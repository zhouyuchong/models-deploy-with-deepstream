import sys
sys.path.append('../')
import time
import torch
from loguru import logger
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst

from common.is_aarch_64 import is_aarch64
from common.bus_call import bus_call
from common.utils import *
from common.display import *
from common.FPS import PERF_DATA

import pyds

import ctypes
ctypes.cdll.LoadLibrary('./models/retinaface/libplugin_rface.so')

conf_thres = 0.25
iou_thres = 0.35

NETWORK_HEIGHT = 640
NETWORK_WIDTH = 640
OSD_PROCESS_MODE= 0
OSD_DISPLAY_TEXT= 1

emotions = ("anger","contempt","disgust","fear","happy","neutral","sad","surprise")
# ----------------------------------------------------------------------------------------------------

# Setting for DeepStream -----------------------------------------------------------------------------
MAX_DISPLAY_LEN = 64
MAX_TIME_STAMP_LEN = 32
MUXER_OUTPUT_WIDTH = 1280  # stream input
MUXER_OUTPUT_HEIGHT = 720  # stream input
# MUXER_BATCH_TIMEOUT_USEC = 4000000
TILED_OUTPUT_WIDTH = 1280 # stream output
TILED_OUTPUT_HEIGHT = 720 # stream output
DRAW_LMKS_SIGNAL = 1
data_type_map = {pyds.NvDsInferDataType.FLOAT: ctypes.c_float,
                    pyds.NvDsInferDataType.INT8: ctypes.c_int8,
                    pyds.NvDsInferDataType.INT32: ctypes.c_int32}
perf_data = None


def pose_src_pad_buffer_probe(pad, info, u_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))

    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number = frame_meta.frame_num
        num_rects = frame_meta.num_obj_meta
        pad_index = frame_meta.pad_index
        l_usr = frame_meta.frame_user_meta_list

        while l_usr is not None:
            try:
                # Casting l_obj.data to pyds.NvDsUserMeta
                user_meta = pyds.NvDsUserMeta.cast(l_usr.data)
            except StopIteration:
                break

            # get tensor output
            if (user_meta.base_meta.meta_type !=
                    pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META):  # NVDSINFER_TENSOR_OUTPUT_META
                try:
                    l_usr = l_usr.next
                except StopIteration:
                    break
                continue

            try:
                tensor_meta = pyds.NvDsInferTensorMeta.cast(
                    user_meta.user_meta_data)
                
                layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)
          
                if tensor_meta.num_output_layers == 1 and layer.layerName == 'output0':
 
                    # layer_output_info = layers_info[0]
                    layer_output_info = pyds.get_nvds_LayerInfo(tensor_meta, 0)  # as num_output_layers == 1

                    network_info = tensor_meta.network_info
                    input_shape = (network_info.width, network_info.height)
                    
                    if frame_number == 0 :
                        print(f'\tmodel input_shape  : {input_shape}')
                    # print("Network Input : w=%d, h=%d, c=%d"%(network_info.width, network_info.height, network_info.channels))

                    # remove zeros from both ends of the array. 'b' : 'both'
                    dims = np.trim_zeros(layer_output_info.dims.d, 'b')
                    
                    if frame_number == 0 :
                        print(f'\tModel output dimension from LayerInfo: {dims}')

                        output_message = f'\tCheck model output shape: {layer_output_info.dims.numElements}, '
                        output_message += f'given OUT_SHAPE : {dims}'
                        assert layer_output_info.dims.numElements == np.prod(dims), output_message
                        
                    
                    # load float* buffer to python
                    cdata_type = data_type_map[layer_output_info.dataType]
                    ptr = ctypes.cast(pyds.get_ptr(layer_output_info.buffer), 
                                    ctypes.POINTER(cdata_type))
                    # Determine the size of the array
                    out = np.ctypeslib.as_array(ptr, shape=dims)

                    if frame_number == 0 :
                        print(f'\tLoad Model Output From LayerInfo. Output Shape : {out.shape}')
                        
                    # [Optional] Postprocess for YOLOv7-pose(with YoloLayer_TRT_v7.0 Layer) prediction tensor
                    # (https://github.com/nanmi/yolov7-pose/)
                    # (57001, 1, 1) > (57000, 1, 1) > (1000, 57)。
                    # out = out[1:, ...].reshape(-1 , 57)   # or out.squeeze()[1:].reshape(-1 , 57)
                    # ----------------------------------------------------------------------------------------------------------------------

                    #  Explicitly specify batch dimensions
                    if np.ndim(out) < 3:
                        out = out[np.newaxis, :]
                        # print(f'add axis 0 for model output : {out.shape}')

                    # [Optional] Postprocess for yolov8s-pose prediction tensor
                    # (https://github.com/triple-Mu/YOLOv8-TensorRT/tree/triplemu/pose-infer)
                    # 　(batch, 56, 8400)　＞(batch, 8400, 56) for yolov8
                    out = out.transpose((0, 2, 1))
                    # # make pseudo class prob
                    cls_prob = np.ones((out.shape[0], out.shape[1], 1), dtype=np.uint8)
                    out[..., :4] = map_to_zero_one(out[..., :4])  # scalar prob to [0, 1]
                    # insert pseudo class prob into predictions
                    out = np.concatenate((out[..., :5], cls_prob, out[..., 5:]), axis=-1)
                    out[..., [0, 2]] = out[..., [0, 2]] * network_info.width  # scale to screen width
                    out[..., [1, 3]] = out[..., [1, 3]] * network_info.height  # scale to screen height
                    # ----------------------------------------------------------------------------------------------------------------------

                    output_shape = (MUXER_OUTPUT_HEIGHT, MUXER_OUTPUT_WIDTH)
                    if frame_number == 0 :
                        print(f'\tModel output : {out.shape}, The coordinates of the keypoint are rescaled to (h, w) : {output_shape}')
                    pred = postprocess(out, output_shape, input_shape,
                                    conf_thres=conf_thres, iou_thres=iou_thres)
                    boxes, confs, kpts = pred
                    if len(boxes) > 0 and len(confs) > 0 and len(kpts) > 0:
                        # add_obj_meta(frame_meta, batch_meta, boxes[0], confs[0])
                        dispaly_frame_pose(frame_meta,
                                        boxes[0], confs[0], kpts[0])


            except StopIteration:
                break

            try:
                l_usr = l_usr.next
            except StopIteration:
                break


        # update frame rate through this probe
        stream_index = "stream{0}".format(frame_meta.pad_index)
        global perf_data
        perf_data.update_fps(stream_index)

        try:
            # indicate inference is performed on the frame
            frame_meta.bInferDone = True
            l_frame = l_frame.next
        except StopIteration:
            break
        # pyds.nvds_acquire_meta_lock(batch_meta)
        # frame_meta.bInferDone = True
        # pyds.nvds_release_meta_lock(batch_meta)

    return Gst.PadProbeReturn.OK

def hand_src_pad_buffer_probe(pad, info, u_data):
    t = time.time()

    frame_number = 0
    num_rects = 0

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))

    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number = frame_meta.frame_num
        num_rects = frame_meta.num_obj_meta
        pad_index = frame_meta.pad_index
        l_usr = frame_meta.frame_user_meta_list

        while l_usr is not None:
            try:
                # Casting l_obj.data to pyds.NvDsUserMeta
                user_meta = pyds.NvDsUserMeta.cast(l_usr.data)
            except StopIteration:
                break

            # get tensor output
            if (user_meta.base_meta.meta_type !=
                    pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META):  # NVDSINFER_TENSOR_OUTPUT_META
                try:
                    l_usr = l_usr.next
                except StopIteration:
                    break
                continue

            try:
                tensor_meta = pyds.NvDsInferTensorMeta.cast(
                    user_meta.user_meta_data)

                # layers_info = []
                for i in range(tensor_meta.num_output_layers):
                    layer = pyds.get_nvds_LayerInfo(tensor_meta, i)
                    print(i, layer.layerName)
                    # layers_info.append(layer)
                    
                # assert tensor_meta.num_output_layers == 2, f'Check number of model output layer : {tensor_meta.num_output_layers}'
                if tensor_meta.num_output_layers == 4:
                # layer_output_info = layers_info[0]
                    layer_output_info = pyds.get_nvds_LayerInfo(tensor_meta, 3)  # as num_output_layers == 1

                    network_info = tensor_meta.network_info
                    input_shape = (network_info.width, network_info.height)

                    # remove zeros from both ends of the array. 'b' : 'both'
                    # print(layer_output_info.dims)
                    dims = np.trim_zeros(layer_output_info.dims.d, 'b')
                
                    # load float* buffer to python
                    cdata_type = data_type_map[layer_output_info.dataType]
                    ptr = ctypes.cast(pyds.get_ptr(layer_output_info.buffer), 
                                    ctypes.POINTER(cdata_type))
                    # Determine the size of the array
                    out = np.ctypeslib.as_array(ptr, shape=dims)
                    out = torch.tensor(out)
                    out = out.reshape(1, 10647, 6).cuda()
                    pred = HandPostProcess.non_max_suppression(out, 0.2, 0.35, classes=None, agnostic=False)

                    if frame_number == 0 :
                        print(f'\tmodel input_shape  : {input_shape}')
                        print(dims)
                    #     print(layer_output_info_2.dims)
                        print("Network Input : w=%d, h=%d, c=%d"%(network_info.width, network_info.height, network_info.channels))
                        
                    # cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
                    # counts, objects, peaks = parse_objects(cmap, paf)


                # # remove zeros from both ends of the array. 'b' : 'both'
                # # print(layer_output_info.dims)
                # dims = np.trim_zeros(layer_output_info.dims.d, 'b')
                
                # if frame_number == 0 :
                #     print(f'\tModel output dimension from LayerInfo: {dims}')
                #     output_message = f'\tCheck model output shape: {layer_output_info.dims.numElements}, '
                #     output_message += f'given OUT_SHAPE : {dims}'
                    
                
                # load float* buffer to python


            except StopIteration:
                break

            try:
                l_usr = l_usr.next
            except StopIteration:
                break


        try:
            # indicate inference is performed on the frame
            frame_meta.bInferDone = True
            l_frame = l_frame.next
        except StopIteration:
            break
        # pyds.nvds_acquire_meta_lock(batch_meta)
        # frame_meta.bInferDone = True
        # pyds.nvds_release_meta_lock(batch_meta)

    return Gst.PadProbeReturn.OK

def osd_sink_pad_buffer_probe(pad, info, u_data):
    if not u_data[1]:
        return Gst.PadProbeReturn.OK	
    scale_ratio = u_data[0]
    frame_number=0
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))    
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break
        
        frame_number=frame_meta.frame_num
        result_landmark = []
        l_user=frame_meta.frame_user_meta_list
        while l_user is not None:
            try:
                user_meta=pyds.NvDsUserMeta.cast(l_user.data) 
            except StopIteration:
                break
            
            if(user_meta and user_meta.base_meta.meta_type==pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META): 
                try:
                    tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
                except StopIteration:
                    break
                

                layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)
                if tensor_meta.num_output_layers == 1 and layer.layerName == 'prob':
                    result_landmark = parse_objects_from_tensor_meta(layer)
                    # print(len(result_landmark))
                   
            try:
                l_user=l_user.next
            except StopIteration:
                break    
          
        num_rects = frame_meta.num_obj_meta
        face_count = 0
        l_obj=frame_meta.obj_meta_list

        # draw 5 landmarks for each rect
        # display_meta.num_circles = len(result_landmark) * 5
        display_meta=pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        ccount = 0
        for i in range(len(result_landmark)):
            # scale coordinates
            landmarks = result_landmark[i] * scale_ratio
            # nvosd struct can only draw MAX 16 elements once 
            # so acquire a new display meta for every face detected
            display_meta=pyds.nvds_acquire_display_meta_from_pool(batch_meta)   
            display_meta.num_rects = 5
            ccount = 0
            for j in range(5):
                py_nvosd_circle_params = display_meta.rect_params[ccount]
                py_nvosd_circle_params.border_color.set(1.0, 1.0, 1.0, 0.3)
                py_nvosd_circle_params.border_width = 5
                py_nvosd_circle_params.has_bg_color = 0
                py_nvosd_circle_params.bg_color.set(0.0, 0.0, 0.0, 1.0)
                py_nvosd_circle_params.left = int(landmarks[j * 2] - 10) if (int(landmarks[j * 2])-10) > 0 else 0
                py_nvosd_circle_params.top = int(landmarks[j * 2 + 1] - 10) if (int(landmarks[j * 2 + 1]) - 10) > 0 else 0
                py_nvosd_circle_params.width = 20
                py_nvosd_circle_params.height = 20
                ccount = ccount + 1
            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

        try:
            l_frame=l_frame.next
        except StopIteration:
            break
			
    return Gst.PadProbeReturn.OK

def genderage_parse_probe(pad, info, u_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))    
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break
    
        l_obj=frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
                
            except StopIteration:
                break

            if obj_meta.unique_component_id == 3:
                l_user_meta = obj_meta.obj_user_meta_list
                while l_user_meta:
                    try:
                        user_meta = pyds.NvDsUserMeta.cast(l_user_meta.data) 
                    except StopIteration:
                        break
                    if user_meta and user_meta.base_meta.meta_type==pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META: 
                        try:
                            tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
                        except StopIteration:
                            break
                
                        layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)
                        # layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)
          
                        if layer.layerName == 'fc1':
                            output = []
                            for i in range(3):
                                output.append(pyds.get_detections(layer.buffer, i))
                            
                            # we don't need gender yet
                            gender_index = int(np.argmax(output[:2]))
                            if gender_index == 1:
                                r,g,b = 0, 0, 255
                                gender = 'Male'
                            else:
                                r,g,b = 255, 129, 190
                                gender = 'Female'
            
                            age = int(np.round(output[2]*100))
                            display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)

                            conf_text_params = display_meta.text_params[0]
                            conf_text_params.display_text = f"age:{age}"
                            if obj_meta.rect_params.left >= TILED_OUTPUT_WIDTH/2:
                                conf_text_params.x_offset = max(int(obj_meta.rect_params.left - 90), 0)
                                conf_text_params.y_offset = max(int(obj_meta.rect_params.top), 0)
                            else:
                                conf_text_params.x_offset = int(obj_meta.rect_params.left + obj_meta.rect_params.width)
                                conf_text_params.y_offset = max(int(obj_meta.rect_params.top), 0)
                            conf_text_params.font_params.font_name = "Serif"
                            conf_text_params.font_params.font_color.set(1.0-r/255, 1.0-g/255, 1.0-b/255, 1.0)
                            conf_text_params.font_params.font_size = 15
                            conf_text_params.set_bg_clr = 1
                            conf_text_params.text_bg_clr.set(r/255, g/255, b/255, 0.5)

                            conf_text_params = display_meta.text_params[1]
                            conf_text_params.display_text = f"{gender}"
                            if obj_meta.rect_params.left >= TILED_OUTPUT_WIDTH/2:
                                conf_text_params.x_offset = max(int(obj_meta.rect_params.left - 90), 0)
                                conf_text_params.y_offset = max(int(obj_meta.rect_params.top + 30), 0)
                            else:
                                conf_text_params.x_offset = int(obj_meta.rect_params.left + obj_meta.rect_params.width)
                                conf_text_params.y_offset = max(int(obj_meta.rect_params.top + 50), 0)
                            conf_text_params.font_params.font_name = "Serif"
                            conf_text_params.font_params.font_color.set(1.0-r/255, 1.0-g/255, 1.0-b/255, 1.0)
                            conf_text_params.font_params.font_size = 15
                            conf_text_params.set_bg_clr = 1
                            conf_text_params.text_bg_clr.set(r/255, g/255, b/255, 0.5)


                            display_meta.num_labels = 2
                            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

                        try:
                            l_user_meta = l_user_meta.next
                        except StopIteration:
                            break                
            try: 
                l_obj=l_obj.next

            except StopIteration:
                break  
        try:
            l_frame=l_frame.next
        except StopIteration:
            break
            
    return Gst.PadProbeReturn.OK	

def emotion_parse_probe(pad, info, u_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))    
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break
    
        l_obj=frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
                
            except StopIteration:
                break

            if obj_meta.unique_component_id == 3:
                l_user_meta = obj_meta.obj_user_meta_list
                while l_user_meta:
                    try:
                        user_meta = pyds.NvDsUserMeta.cast(l_user_meta.data) 
                    except StopIteration:
                        break
                    if user_meta and user_meta.base_meta.meta_type==pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META: 
                        try:
                            tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
                        except StopIteration:
                            break
                
                        layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)
                        if layer.layerName == 'output_emotion':
                            output = []
                            for i in range(8):
                                output.append(pyds.get_detections(layer.buffer, i))

                            emotion_score = round(max(output) * 100, 1)
                            emotion_index = output.index(max(output))
                            # emotions = ("anger","contempt","disgust","fear","happy","neutral","sad","surprise")
                            if emotion_index == 0:
                                r,g,b = 0, 0, 0
                            elif emotion_index == 1:
                                r,g,b = 238, 130, 238
                            elif emotion_index == 2:
                                r,g,b = 106, 90, 205
                            elif emotion_index == 3:
                                r,g,b = 255, 165, 0
                            elif emotion_index == 4:
                                r,g,b = 60, 179, 113
                            elif emotion_index == 5:
                                r,g,b = 72, 111, 63
                            elif emotion_index == 6:
                                r,g,b = 0, 0, 255
                            elif emotion_index == 7:
                                r,g,b = 255, 0, 0

                            obj_meta.rect_params.border_color.set(r/255, g/255, b/255, 1.0)
                            obj_meta.rect_params.border_width = 5
                            obj_meta.rect_params.has_bg_color = 1
                            obj_meta.rect_params.bg_color.set(r/255, g/255, b/255, 0.2)

                            emotion_result = emotions[emotion_index]
                            display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
                            conf_text_params = display_meta.text_params[0]
                            conf_text_params.display_text = f"{emotion_result} ({emotion_score}%)"
                            conf_text_params.x_offset = int(obj_meta.rect_params.left + 10)
                            conf_text_params.y_offset = min(int(obj_meta.rect_params.top + obj_meta.rect_params.height-5), 720)
                            conf_text_params.font_params.font_name = "Serif"
                            conf_text_params.font_params.font_color.set(1.0-r/255, 1.0-g/255, 1.0-b/255, 1.0)
                            conf_text_params.font_params.font_size = 15
                            conf_text_params.set_bg_clr = 1
                            conf_text_params.text_bg_clr.set(r/255, g/255, b/255, 0.5)
                            display_meta.num_labels = 1
                            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

                        try:
                            l_user_meta = l_user_meta.next
                        except StopIteration:
                            break                
            try: 
                l_obj=l_obj.next

            except StopIteration:
                break  
        try:
            l_frame=l_frame.next
        except StopIteration:
            break
            
    return Gst.PadProbeReturn.OK	

def main(args):
    # Check input arguments
    if len(args) != 2:
        sys.stderr.write("usage: %s <v4l2-device-path>\n" % args[0])
        sys.exit(1)

    global perf_data
    perf_data = PERF_DATA(1)

    # Standard GStreamer initialization
    Gst.init(None)

    # Create gstreamer elements
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")

    # Source element for reading from the file
    print("Creating Source \n ")
    source = Gst.ElementFactory.make("v4l2src", "usb-cam-source")
    if not source:
        sys.stderr.write(" Unable to create Source \n")

    caps_v4l2src = Gst.ElementFactory.make("capsfilter", "v4l2src_caps")
    if not caps_v4l2src:
        sys.stderr.write(" Unable to create v4l2src capsfilter \n")


    print("Creating Video Converter \n")

    # Adding videoconvert -> nvvideoconvert as not all
    # raw formats are supported by nvvideoconvert;
    # Say YUYV is unsupported - which is the common
    # raw format for many logi usb cams
    # In case we have a camera with raw format supported in
    # nvvideoconvert, GStreamer plugins' capability negotiation
    # shall be intelligent enough to reduce compute by
    # videoconvert doing passthrough (TODO we need to confirm this)

    # videoconvert to make sure a superset of raw formats are supported
    vidconvsrc = Gst.ElementFactory.make("videoconvert", "convertor_src1")
    # nvvideoconvert to convert incoming raw buffers to NVMM Mem (NvBufSurface API)
    nvvidconvsrc = Gst.ElementFactory.make("nvvideoconvert", "convertor_src2")
    caps_vidconvsrc = Gst.ElementFactory.make("capsfilter", "nvmm_caps")

    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")

    gie_yolo_pose = Gst.ElementFactory.make("nvinfer", "gie_yolo_pose")
    gie_retinaface = Gst.ElementFactory.make("nvinfer", "gie_retinaface")
    gie_face_attr = Gst.ElementFactory.make("nvinfer", "gie_face_attr")
    gie_emotion = Gst.ElementFactory.make("nvinfer", "gie_emotion")

    tiler=Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")

    nvosd_2 = Gst.ElementFactory.make("nvdsosd", "onscreendisplay-2")

    nvosd.set_property('process-mode',OSD_PROCESS_MODE)
    nvosd.set_property('display-text',OSD_DISPLAY_TEXT)
    sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")

    queue1 = Gst.ElementFactory.make("queue", "queue1")
    queue2 = Gst.ElementFactory.make("queue", "queue2")
    queue3 = Gst.ElementFactory.make("queue", "queue3")
    queue4 = Gst.ElementFactory.make("queue", "queue4")
    pipeline.add(queue1)
    pipeline.add(queue2)
    pipeline.add(queue3)
    pipeline.add(queue4)
    # pipeline

    print("Playing cam %s " %args[1])
    caps_v4l2src.set_property('caps', Gst.Caps.from_string("video/x-raw, width=800, height=448, framerate=30/1"))
    caps_vidconvsrc.set_property('caps', Gst.Caps.from_string("video/x-raw(memory:NVMM), format=NV12"))
    source.set_property('device', args[1])
    streammux.set_property('width', MUXER_OUTPUT_WIDTH)
    streammux.set_property('height', MUXER_OUTPUT_HEIGHT)
    streammux.set_property('batch-size', 1)
    streammux.set_property('batched-push-timeout', 4000000)

    gie_yolo_pose.set_property('config-file-path', "configs/config_yolov8_pose.txt")
    gie_retinaface.set_property('config-file-path', "configs/config_face.txt")
    gie_face_attr.set_property('config-file-path', "configs/config_genderage.txt")
    gie_emotion.set_property('config-file-path', "configs/config_emotion.txt")

    tiler.set_property("rows",1)
    tiler.set_property("columns",1)
    tiler.set_property("width", TILED_OUTPUT_WIDTH)
    tiler.set_property("height", TILED_OUTPUT_HEIGHT)
    # Set sync = false to avoid late frame drops at the display-sink
    sink.set_property('sync', False)

    print("Adding elements to Pipeline \n")
    pipeline.add(source)
    pipeline.add(caps_v4l2src)
    pipeline.add(vidconvsrc)
    pipeline.add(nvvidconvsrc)
    pipeline.add(caps_vidconvsrc)
    pipeline.add(streammux)
    pipeline.add(gie_yolo_pose)
    pipeline.add(gie_retinaface)
    pipeline.add(gie_face_attr)
    pipeline.add(gie_emotion)
    pipeline.add(tiler)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    pipeline.add(nvosd_2)
    pipeline.add(sink)

    # we link the elements together
    # v4l2src -> nvvideoconvert -> mux -> 
    # nvinfer -> nvvideoconvert -> nvosd -> video-renderer
    print("Linking elements in the Pipeline \n")
    source.link(caps_v4l2src)
    caps_v4l2src.link(vidconvsrc)
    vidconvsrc.link(nvvidconvsrc)
    nvvidconvsrc.link(caps_vidconvsrc)

    sinkpad = streammux.get_request_pad("sink_0")
    if not sinkpad:
        sys.stderr.write(" Unable to get the sink pad of streammux \n")
    srcpad = caps_vidconvsrc.get_static_pad("src")
    if not srcpad:
        sys.stderr.write(" Unable to get source pad of caps_vidconvsrc \n")
    srcpad.link(sinkpad)
    streammux.link(gie_yolo_pose)
    gie_yolo_pose.link(queue1)
    queue1.link(gie_retinaface)
    gie_retinaface.link(queue2)
    queue2.link(gie_face_attr)
    gie_face_attr.link(queue3)
    queue3.link(gie_emotion)
    gie_emotion.link(queue4)
    queue4.link(tiler)
    tiler.link(nvvidconv)
    nvvidconv.link(nvosd)
    nvosd.link(sink)


    # create an event loop and feed gstreamer bus mesages to it
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect ("message", bus_call, loop)

    scale_ratio = cal_ratio(NETWORK_HEIGHT, NETWORK_WIDTH, TILED_OUTPUT_HEIGHT, TILED_OUTPUT_WIDTH)
    user_data = [scale_ratio, DRAW_LMKS_SIGNAL]

    # Lets add probe to get informed of the meta data generated, we add probe to
    # the sink pad of the osd element, since by that time, the buffer would have
    # had got all the metadata.
    # osdsinkpad = nvosd.get_static_pad("sink")
    # if not osdsinkpad:
    #     sys.stderr.write(" Unable to get sink pad of nvosd \n")

    # osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)
    pgie_src_pad = queue1.get_static_pad("sink")
    pgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, pose_src_pad_buffer_probe, 0)


    # sgie_src_pad = queue2.get_static_pad("sink")
    # sgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, hand_src_pad_buffer_probe, 0)

    tgie_src_pad = queue2.get_static_pad("sink")
    tgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, user_data)

    genderage_parse_pad = queue3.get_static_pad("sink")
    genderage_parse_pad.add_probe(Gst.PadProbeType.BUFFER, genderage_parse_probe, 0)

    emotion_parse_pad = queue4.get_static_pad("sink")
    emotion_parse_pad.add_probe(Gst.PadProbeType.BUFFER, emotion_parse_probe, 0)

    GLib.timeout_add(5000, perf_data.perf_print_callback)

    # start play back and listen to events
    print("Starting pipeline \n")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    # cleanup
    pipeline.set_state(Gst.State.NULL)

if __name__ == '__main__':
    sys.exit(main(sys.argv))


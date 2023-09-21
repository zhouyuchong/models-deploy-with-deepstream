/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <algorithm>
#include <map>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <unordered_map>

#include <iomanip>

#include "nvdsinfer_custom_impl.h"

static const int COCO_NAMES = 80;
#define NMS_THRESH 0.5
#define CONF_THRESH 0.25
#define VIS_THRESH 0.5
#define BATCH_SIZE 1
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define CLIP(a, min, max) (MAX(MIN(a, max), min))

extern "C" bool NvDsInferParseCustomFastestDet(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferParseObjectInfo> &objectList);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
static constexpr int LOCATIONS = 4;

inline float sigmoid(float x)
{
	return 1.0 / (1 + expf(-x));
}

struct alignas(float) Detection{
        float bbox[LOCATIONS];
        float conf;  // bbox_conf * cls_conf
        float class_id;
    };

float iou(float lbox[4], float rbox[4]) {
    float interBox[] = {
        std::max(lbox[0] - lbox[2]/2.f , rbox[0] - rbox[2]/2.f), //left
        std::min(lbox[0] + lbox[2]/2.f , rbox[0] + rbox[2]/2.f), //right
        std::max(lbox[1] - lbox[3]/2.f , rbox[1] - rbox[3]/2.f), //top
        std::min(lbox[1] + lbox[3]/2.f , rbox[1] + rbox[3]/2.f), //bottom
    };

    if(interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS =(interBox[1]-interBox[0])*(interBox[3]-interBox[2]);
    return interBoxS/(lbox[2]*lbox[3] + rbox[2]*rbox[3] -interBoxS);
}

bool cmp(Detection& a, Detection& b) {
    return a.conf > b.conf;
}

void nms_and_adapt(std::vector<Detection>& det, std::vector<Detection>& res, float nms_thresh, int width, int height) {
    std::sort(det.begin(), det.end(), cmp);
    for (unsigned int m = 0; m < det.size(); ++m) {
        auto& item = det[m];
        res.push_back(item);
        for (unsigned int n = m + 1; n < det.size(); ++n) {
            if (iou(item.bbox, det[n].bbox) > nms_thresh) {
                det.erase(det.begin()+n);
                --n;
            }
        }
    }

    for (unsigned int m = 0; m < res.size(); ++m) {
        res[m].bbox[0] = CLIP(res[m].bbox[0], 0, width - 1);
        res[m].bbox[1] = CLIP(res[m].bbox[1], 0, height -1);
        res[m].bbox[2] = CLIP(res[m].bbox[2], 0, width - 1);
        res[m].bbox[3] = CLIP(res[m].bbox[3], 0, height - 1);
    }

}

void decode_output(float *output, std::vector<Detection>& temp, int num_grid_x, int num_grid_y, int labelStartIndex, int dimensions, int width, int height) {
    //reshape raw output
    float data[22][22][85];
    int index = 0;
    for (int z=0; z<85; z++) {
        for (int y=0;y<num_grid_y;y++) {
            for (int x=0;x<num_grid_x; x++) {
                data[y][x][z] = output[index];
                index++;
            }
        }
    }

    for (int i = 0; i < num_grid_y; i++) {
		for (int j = 0; j < num_grid_x; j++) {
            
            // first get confidence for filter
            float conf = data[i][j][0];
            if (conf >= CONF_THRESH) {
                // get label class id
                float max = 0;
                int token = 0;
                for (int k = labelStartIndex; k < dimensions; k++) {
                    if (data[i][j][k] > max) {
                        max = data[i][j][k];
                        token = k;
                    }
                }

                int class_idx = token - 5;
                // get bbox
                float bcx = data[i][j][1];
                float bcy = data[i][j][2];
                float bw = data[i][j][3];
                float bh = data[i][j][4];
                
                bcx = (tanh(bcx) + j) / (float)num_grid_x;
                bcy = (tanh(bcy) + i) / (float)num_grid_y;
                bw = sigmoid(bw);
                bh = sigmoid(bh);
                
                float left = bcx - 0.5 * bw;
                float top = bcy - 0.5 * bh;

                Detection det;
                det.conf = conf;
                det.bbox[0] = left * width;//top left x
                det.bbox[1] = top * height;//top left y
                det.bbox[2] = int(bw * width);//width
                det.bbox[3] = int(bh * height);//height
                det.class_id = class_idx;
                temp.push_back(det); 
            }
		}
	}


}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/* This is a sample bounding box parsing function for the sample FastestDet detector model */
static bool NvDsInferParseFastestDet(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    std::cout<<detectionParams
    std::vector<Detection> temp;
    std::vector<Detection> res;

    int dimensions = 85; // 0 -> confidence ,1,2,3,4 -> box, 5-85 -> coco classes confidence 
    int labelStartIndex = 5;
    const int num_grid_x = 22;
	const int num_grid_y = 22;

    decode_output((float*)(outputLayersInfo[0].buffer), temp, num_grid_x, num_grid_y, labelStartIndex, dimensions, networkInfo.width, networkInfo.height);
    nms_and_adapt(temp, res, NMS_THRESH, networkInfo.width, networkInfo.height);
    
    for(auto& r : res) {
        
        if(r.conf<=VIS_THRESH) continue;

	    NvDsInferParseObjectInfo oinfo;  
	    oinfo.classId = r.class_id;
	    oinfo.left    = static_cast<unsigned int>(r.bbox[0]);
	    oinfo.top     = static_cast<unsigned int>(r.bbox[1]);
	    oinfo.width   = static_cast<unsigned int>(r.bbox[2]);
	    oinfo.height  = static_cast<unsigned int>(r.bbox[3]);
	    oinfo.detectionConfidence = r.conf;
        objectList.push_back(oinfo);
             
    }

    return true;   
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/* C-linkage to prevent name-mangling */
extern "C" bool NvDsInferParseCustomFastestDet(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferParseObjectInfo> &objectList)
{
    return NvDsInferParseFastestDet(
        outputLayersInfo, networkInfo, detectionParams, objectList);
}


/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomFastestDet);

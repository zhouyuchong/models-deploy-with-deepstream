#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include "nvdsinfer_custom_impl.h"

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define CLIP(a,min,max) (MAX(MIN(a, max), min))
#define DIVIDE_AND_ROUND_UP(a, b) ((a + b - 1) / b)

static bool dict_ready=false;
static const int NUM_CLASSES_YOLO = 80;


/* YOLOv4 TLT from detection to recognition*/
extern "C" bool NvDsInferParseCustomGenderAge(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                 NvDsInferNetworkInfo const &networkInfo, float classifierThreshold,
                                 std::vector<NvDsInferAttribute> &attrList, std::string &attrString) {

    //  if(outputLayersInfo.size() != 4)
    // {
    //     std::cerr << "Mismatch in the number of output buffers."
    //               << "Expected 4 output buffers, detected in the network :"
    //               << outputLayersInfo.size() << std::endl;
    //     return false;
    // }


    float* result = (float *) outputLayersInfo[0].buffer;
    NvDsInferDims dim = outputLayersInfo[0].dims;
    // std::cout<<outputLayersInfo[0].layerName<<" "<<dim.numElements<<std::endl;
    int index = dim.numElements;
    

    // float* p_bboxes = (float *) outputLayersInfo[1].buffer;
    // float* p_scores = (float *) outputLayersInfo[2].buffer;
    // float* p_classes = (float *) outputLayersInfo[3].buffer;

    const int max_length = 10;
    // std::cout<<result.layerName<<std::endl;

    char *gender;
    if (result[0] > result[1]) {
        gender = "male";
    } else {
        gender = "female";
    }
    int age = ceil(result[2]*100);
    std::cout<<"gender: "<<gender<<"  age: "<<age<<std::endl;

    NvDsInferAttribute LPR_attr;


    LPR_attr.attributeIndex = 0;
    LPR_attr.attributeValue = 1;
    LPR_attr.attributeLabel = strdup(attrString.c_str());
    for (unsigned int count = 0; count < valid_bank_count; count++) {
        LPR_attr.attributeConfidence *= bank_softmax_max[count];
    }
    attrList.push_back(LPR_attr);
    // const char* log_enable = std::getenv("ENABLE_DEBUG");

    // if(log_enable != NULL && std::stoi(log_enable)) {
    //     std::cout <<"keep cout"
    //           <<p_keep_count[0] << std::endl;
    // }
    // Read dict
    // std::ifstream fdict;
    // setlocale(LC_CTYPE, "");

    // if(!dict_ready) {
    //     fdict.open("dict.txt");
    //     if(!fdict.is_open())
    //     {
    //         std::cout << "open dictionary file failed." << std::endl;
    //         return false;
    //     }
    //     while(!fdict.eof()) {
    //         std::string strLineAnsi;
    //         if (getline(fdict, strLineAnsi) ) {
    //             if (strLineAnsi.length() > 1) {
    //                 strLineAnsi.erase(1);
    //             }
    //             dict_table.push_back(strLineAnsi);
    //         }
    //     }
    //     dict_ready=true;
    //     fdict.close();
    // }

    // Empty list of object
    std::vector <ObjectPoint> objectList;
    // Append detection result
    objectList.size() <= max_length
    for (int i = 0; i < 10 ; i++) {

        // if ( (float)p_scores[i] < classifierThreshold) continue;

        // if(log_enable != NULL && std::stoi(log_enable)) {
        //     std::cout << "label/conf/ x/y x/y -- "
        //               << p_classes[i] << " " << p_scores[i] << " "
        //               << p_bboxes[4*i] << " " << p_bboxes[4*i+1] << " " << p_bboxes[4*i+2] << " "<< p_bboxes[4*i+3] << " " << std::endl;
        // }

        if(p_bboxes[4*i+2] < p_bboxes[4*i] || p_bboxes[4*i+3] < p_bboxes[4*i+1]) continue;

        ObjectPoint obj;
        obj.ctx = (float)(p_bboxes[4*i] + p_bboxes[4*i+2])/2.0;
        obj.cty = (float)(p_bboxes[4*i+1] + p_bboxes[4*i+3])/2.0;
        obj.width = (float)p_bboxes[4*i+2] - p_bboxes[4*i];
        obj.height = (float)p_bboxes[4*i+3] - p_bboxes[4*i+1];
        obj.confidence = (float)p_scores[i];
        obj.classId = (int) p_classes[i];

        if(obj.height < 0 || obj.width < 0)
            continue;
        objectList.push_back(obj);
    }
    Add to metadata
    NvDsInferAttribute LPR_attr;
    // LPR_attr.attributeConfidence = sumConfidence / objectListConfidence.size();
    attrString = mergeDetectionResult(objectList);
    if (objectList.size() >=  3) {
        LPR_attr.attributeIndex = 0;
        LPR_attr.attributeValue = 1;
        LPR_attr.attributeConfidence = 1.0;
        LPR_attr.attributeLabel = strdup(attrString.c_str());
        for (ObjectPoint obj: objectList)
            LPR_attr.attributeConfidence *= obj.confidence;
        attrList.push_back(LPR_attr);
        // std::cout << "License plate: " << attrString << "  -  Confidence: " << LPR_attr.attributeConfidence << std::endl;
    }

    return true;
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_CLASSIFIER_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomGenderAge);

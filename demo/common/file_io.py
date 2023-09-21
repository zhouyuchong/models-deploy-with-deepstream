'''
Author: zhouyuchong
Date: 2023-09-01 09:58:09
Description: 
LastEditors: zhouyuchong
LastEditTime: 2023-09-07 14:03:21
'''

import os
import re
from loguru import logger

import numpy as np


def init_analytics_config_file(path):
    '''
    Create an empty analytics config file.
    + Args:
        max_source_number: to write corresponding number of groups.
    '''
    if os.path.exists(path):  # 如果文件存在
        # 删除文件，可使用以下两种方法。
        os.remove(path)
    text_line = []
    text_line.append(
        "[property]\nenable=1\nconfig-width=1080\nconfig-height=720\nosd-mode=2\ndisplay-font-size=12\n\n")
    file = open(path, 'w')
    # if "LINE" in types:
    #     for i in range(max_source_number + 1):
    #         text_line.append(
    #             "[line-crossing-stream-{0}]\nenable=0\nline-crossing-Exit=789;672;1084;900;851;773;1203;732\nextended=0\nmode=balanced\nclass-id=-1\n\n".format(
    #                 i))
    # if "ROI" in types:
    #     for i in range(max_source_number + 1):
    #         text_line.append(
    #             "[roi-filtering-stream-{0}]\nroi-{1}=0;0;0;0;0;0;0;0\nenable=0\ninverse-roi=0\nclass-id=-1\n\n".format(i,
    #                                                                                                                  i))
    # if "DIR" in types:
    #     for i in range(max_source_number + 1):
    #         text_line.append(
    #             "[direction-detection-stream-{0}]\ndirection-{1}=0;0;1;1\nenable=0\nmode=balanced\nclass-id=-1\n\n".format(i,
    #                                                                                                                  i))

    file.writelines(text_line)
    file.close()


def add_analytics_roi(filepath, index, ana_type, ptz_params, MUXER_OUTPUT_WIDTH, MUXER_OUTPUT_HEIGHT):
    logger.info("analytics type contains: {}".format(ana_type))
    logger.debug(ptz_params)
    for single_ana_type in ana_type:
    
        key = '[{}-stream-{}]'.format(single_ana_type, index)
        with open(filepath, mode='a') as file:
            file.write(key)
            file.write('\n')
            # file.write("enable=1\nroi-{}={}\ninverse-roi=0\nclass-id=-1\n\n")

            insert_str = ''
            if single_ana_type == 'roi-filtering':
                if 'coordinate' not in ptz_params:
                    continue
                for coor_name, coor_data in ptz_params['coordinate'].items():
                    # print("name:{} data:{}".format(coor_name, coor_data))
                    coor_data_np = np.array(coor_data.split(';')).astype(np.float64)
                    coor_data_np[::2] *= MUXER_OUTPUT_WIDTH
                    coor_data_np[1::2] *= MUXER_OUTPUT_HEIGHT
                    coor_data_str = ';'.join([str(int(i)) for i in coor_data_np])
                    insert_str = insert_str + 'roi-{}={}\n'.format(coor_name, coor_data_str)
                    # print("insert params: {}".format(insert_str))
                insert_str = insert_str + 'enable=1\ninverse-roi=0\nclass-id=-1\n\n'

            elif single_ana_type == 'direction-detection':
                if 'direction' not in ptz_params:
                    continue

                for coor_name, coor_data in ptz_params['direction'].items():
                    coor_data_np = np.array(coor_data.split(';')).astype(np.float64)
                    coor_data_np[::2] *= MUXER_OUTPUT_WIDTH
                    coor_data_np[1::2] *= MUXER_OUTPUT_HEIGHT
                    coor_data_inverse = [str(int(i)) for i in coor_data_np]
                    coor_data_inverse = ';'.join(coor_data_inverse[2:] + coor_data_inverse[:2])
                    insert_str = insert_str + 'direction-{}={}\n'.format(coor_name, coor_data_inverse)
                insert_str = insert_str + 'enable=1\nmode=balanced\nclass-id=-1\n\n'

            file.write(insert_str)

    return True



def create_subdirs(root_path, ip, ptz_list):
    path = os.path.join(root_path, str(ip)) 
    logger.debug("Creating subdirs to save results, path: {}.".format(path))
    if not os.path.exists(path):
        os.makedirs(path)
    for ptz in ptz_list:
        sub_path = path + '/' + str(ptz)
        if not os.path.exists(sub_path):
            os.makedirs(sub_path)

    return path

# %% package import
import os
import argparse

import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver

from utils.yolo_classes import get_cls_dict
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO

# KAIST import
from typing import final
import numpy as np
import pandas as pd
import math
import time
import json
import ast
from os import listdir
from os.path import isfile, join
import natsort
from numpy.linalg import norm
import cv2

##from load_json import log_read

from func_init_grid import GridArray

# %% Functions


def parse_args():
    desc = ('Run the TensorRT optimized object detecion model on an input '
            'video and save BBoxed overlaid output as another video.')
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        '-v', '--video', type=str, required=True,
        help='input video file name')
    parser.add_argument(
        '-o', '--output', type=str, required=True,
        help='output video file name')
    parser.add_argument(
        '-c', '--category_num', type=int, default=80,
        help='number of object categories [80]')
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help=('[yolov3-tiny|yolov3|yolov3-spp|yolov4-tiny|yolov4|'
              'yolov4-csp|yolov4x-mish]-[{dimension}], where '
              '{dimension} could be either a single number (e.g. '
              '288, 416, 608) or 2 numbers, WxH (e.g. 416x256)'))
    parser.add_argument(
        '-l', '--letter_box', action='store_true',
        help='inference with letterboxed image [False]')
    args = parser.parse_args()
    return args


def car_tracking(obj_info, car_ref_frame_label, car_ref_frame_axies, car_label_cnt):

    global min_distance

    x = int(obj_info[0])
    y = int(obj_info[1])
    lbl = float(obj_info[2])

    b = np.array([(x, y)])
    a = np.array(car_ref_frame_axies)

    distance = norm(a-b, axis=1)

    dist_min_value = distance.min()

    if dist_min_value < min_distance:
        idx = np.where(distance == dist_min_value)[0][0]
        lbl = car_ref_frame_label[idx]

    if math.isnan(lbl):
        lbl = car_label_cnt
        car_label_cnt += 1

    return car_ref_frame_label, car_ref_frame_axies, lbl, car_label_cnt


def ped_tracking(obj_info, person_ref_frame_label, person_ref_frame_axies, person_label_cnt):

    global min_distance

    x = int(obj_info[0])
    y = int(obj_info[1])
    lbl = float(obj_info[2])

    b = np.array([(x, y)])
    a = np.array(person_ref_frame_axies)

    distance = norm(a-b, axis=1)

    dist_min_value = distance.min()

    if dist_min_value < min_distance:
        idx = np.where(distance == dist_min_value)[0][0]
        lbl = person_ref_frame_label[idx]

    if math.isnan(lbl):
        lbl = person_label_cnt
        person_label_cnt += 1

    return person_ref_frame_label, person_ref_frame_axies, lbl, person_label_cnt


def get_pixel_distance(x1, y1, x2, y2):

    return math.sqrt((x1-y1)**2 + (x2-y2)**2)


def get_speed(ref_object_info, cur_object_info):

    # ref_object_info : (x, y)
    # cur_object_info : (x, y)

    global FPS
    global SAMPLEING_RATE
    global K

    pixel_dist = get_pixel_distance(
        ref_object_info[0], ref_object_info[1], cur_object_info[0], cur_object_info[1])

    # calculating with FPS, SAIMPLEING_RATE, and K

    return pixel_dist / FPS


def get_object_behaivor(obj_ref_frame_label, obj_ref_frame_axies, obj_cur_frame_label, obj_cur_frame_axies):

    ref_len = len(obj_ref_frame_label)
    ref_cur = len(obj_cur_frame_label)

    obj_behavior_set = dict()

    speed_set = []

    if ref_len != 0:

        for ref_trace_index in range(0, ref_len):
            ref_object_id = obj_ref_frame_label[ref_trace_index]
            ref_object_info = obj_ref_frame_axies[ref_trace_index]

            obj_match_flag = False

            for cur_trace_index in range(0, ref_cur):
                cur_object_id = obj_cur_frame_label[cur_trace_index]

                if ref_object_id == cur_object_id:
                    obj_match_flag = True

                    cur_object_info = obj_cur_frame_axies[cur_trace_index]

                    # get_speed
                    object_speed = get_speed(
                        ref_object_info, cur_object_info)

                    break

            if not obj_match_flag:
                # print("test")
                # print(object_speed)
                object_speed = 0
                # print("???")

            speed_set.append(object_speed)
            # print(speed_set)

        obj_behavior_set["speed"] = speed_set

    # return obj_behavior_set
    return speed_set


def get_obj_info(obj_ref_frame_label, obj_ref_frame_axies, obj_grid_list, obj_speed_list, obj_traj_dict, obj_grid_traj_dict, obj_speed_dict, frame_index, frame_index_dict):

    for obj_lbl, obj_axies, obj_grid, obj_speed in zip(obj_ref_frame_label, obj_ref_frame_axies, obj_grid_list, obj_speed_list):
        if obj_lbl not in obj_traj_dict.keys():
            # print("Key does not exists")
            obj_traj_dict[obj_lbl] = []
            obj_grid_traj_dict[obj_lbl] = []
            obj_speed_dict[obj_lbl] = []
            frame_index_dict[obj_lbl] = frame_index

        obj_traj_dict[obj_lbl].append(obj_axies)
        obj_grid_traj_dict[obj_lbl].append(obj_grid)
        obj_speed_dict[obj_lbl].append(obj_speed)

    return obj_traj_dict, obj_grid_traj_dict, obj_speed_dict, frame_index_dict


def frame_filtering(frame_index, obj_traj_dict, obj_grid_traj_dict, obj_bhvr_dict, frame_index_dict):

    rs_obj_traj_dict = dict()
    rs_obj_grid_traj_dict = dict()
    rs_obj_bhvr_dict = dict()
    rs_frame_index_dict = dict()

    value_list = []

    # print(frame_index_dict)

    for key, value in frame_index_dict.items():

        if value > frame_index - DIFF_FRAME:
            value_list.append(key)

    # print(value_list)

    rs_obj_traj_dict = {
        key: obj_traj_dict[key] for key in value_list}
    rs_obj_grid_traj_dict = {
        key: obj_grid_traj_dict[key] for key in value_list}
    rs_obj_bhvr_dict = {
        key: obj_bhvr_dict[key] for key in value_list}

    rs_frame_index_dict = {
        key: frame_index_dict[key] for key in value_list}

    return rs_obj_traj_dict, rs_obj_grid_traj_dict, rs_obj_bhvr_dict, rs_frame_index_dict
# %% Grid initialization
# seg_color = [[0, 0, 0], [6, 154, 78], [128, 128, 128], [108, 225, 194]]
# seg_img = cv2.imread('../src/seg_map_v2_short.png')

# k = 15
# shape = (745, 800, 3)

# ga = GridArray(k, shape, seg_img)
# %% Grid class definition in KAIST


class setGrid:

    def __init__(self, num_cells, virtual_img_size):
        self.num_cells = num_cells

        # width == height in all cases
        self.virtual_img_width = virtual_img_size[0] = 1000
        self.virtual_img_height = virtual_img_size[1] = 1000

        self.grid_array = np.full(
            (self.num_cells * self.num_cells), 0, np.uint8)

        self.grid_pixel_ratio = float(self.virtual_img_width / self.num_cells)

        # self.grid_index = 0

    def get_grid_from_coord(self, coords, clss):

        base_boundary = [0, 0, self.grid_pixel_ratio, self.grid_pixel_ratio]
        # print(base_boundary)
        # print(self.grid_pixel_ratio)

        # break_flag = False
        grid_index = 0

        rs_grid_list = []

        for coord, cls in zip(coords, clss):
            # print(coord)
            break_flag = False
            for horizon_shift_index in range(0, self.num_cells):
                target_X = coord[0] - \
                    self.grid_pixel_ratio * horizon_shift_index

                # if target_X < base_boundary[0]:
                #    grid_index = 0
                #    break

                if (base_boundary[0] <= target_X) and (target_X <= base_boundary[2]):

                    for vertical_shift_index in range(0, self.num_cells):
                        target_Y = coord[1] - \
                            self.grid_pixel_ratio * vertical_shift_index

                        if target_Y < base_boundary[1]:
                            grid_index = 0
                            break_flag = True

                        if (base_boundary[1] <= target_Y) and (target_Y <= base_boundary[3]):
                            grid_index = horizon_shift_index + vertical_shift_index * self.num_cells
                            # rs_grid_list.append(grid_index)
                            break_flag = True
                            break

                if break_flag:
                    break

            rs_grid_list.append(grid_index)

        return rs_grid_list


grid = setGrid(100, [1000, 1000])
# function use case
# grid.get_grid_from_coord(xy_coodinates_list, class_list)
grid.get_grid_from_coord(
    [(236, 296), (235, 295), (235, 295), (235, 295)], ['1']*4)


# %% log data loading and extract objects' infromation by frame

FPS = 10
SAMPLING_RATE = 10
K = 10
P = 10
DIFF_FRAME = 30

# log_dir = 'logs_by_frame'
log_dir = 'E:\#4 Experiment source\#6 나비박스\#6 NAVIBOX_window_workspace\logs_by_frame'
# site_1_log_dir = 'yuseong_logs_by_frame'

dat = log_read(log_dir)
# dat = log_read(site_1_log_dir)

len_dat = len(dat)

person_ref_frame_axies = []
person_ref_frame_label = []

car_ref_frame_axies = []
car_ref_frame_label = []

person_label_cnt = 1
car_label_cnt = 1


min_distance = 30
lbl_cnt = 1

# 위험상황 호출 함수 #
SetMode_01 = 1  # 무단횡단
SetMode_02 = 2  # 충돌

RiskLevel_01 = 1
RiskLevel_02 = 2
RiskLevel_03 = 3
# ContextMode ("/", SetMode_02, RiskLevel_01)
# ContextMode ("/", SetMode_02, RiskLevel_02)
# ContextMode ("/", SetMode_02, RiskLevel_03)

# def get_obj_grid(obj_ref_frame_label, obj_ref_frame_axies, obj_grid_list, obj_traj_dict, obj_grid_traj_dict):

#     for obj_lbl, obj_axies, obj_grid in zip(obj_ref_frame_label, obj_ref_frame_axies, obj_grid_list):
#         if obj_lbl not in obj_traj_dict.keys():
#             # print("Key does not exists")
#             obj_traj_dict[obj_lbl] = []
#             obj_grid_traj_dict[obj_lbl] = []

#         obj_traj_dict[obj_lbl].append(obj_axies)
#         obj_grid_traj_dict[obj_lbl].append(obj_grid)

#     return obj_traj_dict, obj_grid_traj_dict

car_traj_dict = dict()
ped_traj_dict = dict()
car_grid_traj_dict = dict()
ped_grid_traj_dict = dict()

car_bhvr_dict = dict()
ped_bhvr_dict = dict()

car_frame_index_dict = dict()
ped_frame_index_dict = dict()


def loop_and_detect(cap, trt_yolo, conf_th, vis, writer):
    while True:
        ret, frame = cap.read()		# input source로부터 한 프레임 read
        if frame is None:
            break
        # 모델로부터 탐지 결과 np.ndarray로 받음 (바운딩박스 좌표, 점수, 클래스)
        boxes, confs, clss = trt_yolo.detect(frame, conf_th)
        # 시각화	# 좌표는 int타입 x_min, y_min, x_max, y_max 순서

        # KAIST SECTION START
        person_cur_frame_axies = []
        person_cur_frame_label = []

        car_cur_frame_axies = []
        car_cur_frame_label = []

        #target_frame = dat[frame_index]
        # target_frame_info = target_frame[str(frame_index+1)]  # outputs

        tmp_integrated_test_list = []
        tmp_integrated_test_dict = dict()

        #boxes = [[1, 2, 3, 4], [5, 6, 7, 8]]
        #clss_ = [0, 1]

        for i in range(0, len(boxes)):

            tmp_integrated_test_dict['min_x'] = boxes[i][0]
            tmp_integrated_test_dict['min_y'] = boxes[i][1]
            tmp_integrated_test_dict['max_x'] = boxes[i][2]
            tmp_integrated_test_dict['max_y'] = boxes[i][3]

            tmp_integrated_test_dict['class'] = clss_[i]

            tmp_integrated_test_list.append(tmp_integrated_test_dict)

        module_AB_start_time = time.time()

        for output in target_frame_info:

            # Module A: Object info. parsing

            left = int(output['min_y'])
            top = int(output['min_x'])
            right = int(output['max_y'])
            bottom = int(output['max_x'])

            clss = output['class']

            lbl = float('nan')

            # Module B: Object tracking
            module_B_start_time = time.time()
            if clss != 1:  # person

                x = int((top+left) / 2.0)
                y = int(right)

                if (len(person_ref_frame_label) > 0):
                    person_ref_frame_label, person_ref_frame_axies, lbl, person_label_cnt = ped_tracking((x, y, lbl),
                                                                                                         person_ref_frame_label,
                                                                                                         person_ref_frame_axies,
                                                                                                         person_label_cnt)

                person_cur_frame_label.append(lbl)
                person_cur_frame_axies.append((x, y))

            else:  # no person

                x = int(left)
                y = int(right)

                if (len(car_ref_frame_label) > 0):
                    car_ref_frame_label, car_ref_frame_axies, lbl, car_label_cnt = car_tracking((x, y, lbl),
                                                                                                car_ref_frame_label,
                                                                                                car_ref_frame_axies,
                                                                                                car_label_cnt)

                car_cur_frame_label.append(lbl)
                car_cur_frame_axies.append((x, y))

        print("Modules A and B Done", time.time() - module_AB_start_time)

        # Module C: Trajectory parsing and behavior info extracting including grid matching
        module_C_start_time = time.time()
        ped_grid_list = grid.get_grid_from_coord(
            person_ref_frame_axies, ['1']*len(person_ref_frame_axies))

        car_grid_list = grid.get_grid_from_coord(
            car_ref_frame_axies, ['1']*len(car_ref_frame_axies))

        ped_bhvr_list = get_object_behaivor(
            person_ref_frame_label, person_ref_frame_axies, person_cur_frame_label, person_cur_frame_axies)

        car_bhvr_list = get_object_behaivor(
            car_ref_frame_label, car_ref_frame_axies, car_cur_frame_label, car_cur_frame_axies)

        ped_traj_dict, ped_grid_traj_dict, ped_bhvr_dict, ped_frame_index_dict = get_obj_info(person_ref_frame_label, person_ref_frame_axies,
                                                                                              ped_grid_list, ped_bhvr_list,
                                                                                              ped_traj_dict, ped_grid_traj_dict, ped_bhvr_dict,
                                                                                              frame_index, ped_frame_index_dict)

        car_traj_dict, car_grid_traj_dict, car_bhvr_dict, car_frame_index_dict = get_obj_info(car_ref_frame_label, car_ref_frame_axies,
                                                                                              car_grid_list, car_bhvr_list,
                                                                                              car_traj_dict, car_grid_traj_dict, car_bhvr_dict,
                                                                                              frame_index, car_frame_index_dict)

        # # Filtering

        ped_traj_dict, ped_grid_traj_dict, ped_bhvr_dict, ped_frame_index_dict = frame_filtering(frame_index,
                                                                                                 ped_traj_dict,
                                                                                                 ped_grid_traj_dict,
                                                                                                 ped_bhvr_dict,
                                                                                                 ped_frame_index_dict)

        car_traj_dict, car_grid_traj_dict, car_bhvr_dict, car_frame_index_dict = frame_filtering(frame_index,
                                                                                                 car_traj_dict,
                                                                                                 car_grid_traj_dict,
                                                                                                 car_bhvr_dict,
                                                                                                 car_frame_index_dict)
        print("Module C Done", time.time() - module_C_start_time)

        # Module D: Potential collision risk prediction
        module_D_start_time = time.time()
        # ped_traj_dict, ped_grid_traj_dict, ped_bhvr_dict
        # car_traj_dict, car_grid_traj_dict, car_bhvr_dict

        def get_PCR(ped_traj_dict, ped_grid_traj_dict, ped_bhvr_dict, car_traj_dict, car_grid_traj_dict, car_bhvr_dict):
            # TBD
            ped_grid_point = list(ped_grid_traj_dict.values())
            car_grid_point = list(car_grid_traj_dict.values())

            for ped_index in range(0, len(ped_grid_point)):
                target_ped = ped_grid_point[ped_index]

                for ped_frame_index in range(0, len(target_ped)):
                    for car_index in range(0, len(car_grid_point)):
                        target_car = car_grid_point[car_index]

                        for car_frame_index in range(0, len(target_car)):

                            if target_car[car_frame_index] == target_ped[ped_frame_index]:
                                ContextMode("/", SetMode_02, RiskLevel_03)

                            else:
                                ContextMode("/", SetMode_02, RiskLevel_02)

                ContextMode("/", SetMode_02, RiskLevel_03)

        get_PCR(ped_traj_dict, ped_grid_traj_dict, ped_bhvr_dict,
                car_traj_dict, car_grid_traj_dict, car_bhvr_dict)

        print("Module D Done", time.time() - module_D_start_time)
        # Module D: Hard memorials and short memorials removing

        # if frame_index == 99:
        #    break

        person_ref_frame_label = person_cur_frame_label
        person_ref_frame_axies = person_cur_frame_axies

        car_ref_frame_label = car_cur_frame_label
        car_ref_frame_axies = car_cur_frame_axies
        # KAIST SECTION END

        frame = vis.draw_bboxes(frame, boxes, confs, clss)
        writer.write(frame)		# 프레임 저장 (영상처리)	# 		# 점수는 float타입
        print('.', end='', flush=True)						# 클래스는 float타입

    print('\nDone.')


# %%

def main():
    # argument parsing ----
    args = parse_args()
    if args.category_num <= 0:
        raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
    if not os.path.isfile('yolo/%s.trt' % args.model):
        raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % args.model)
    # ---------------------

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit('ERROR: failed to open the input video file!')
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))		# 영상
    writer = cv2.VideoWriter(							# 포맷
        args.output,								# 세팅
        cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))  # 부분

    # 클래스 정보 불러오기
    cls_dict = get_cls_dict(args.category_num)
    vis = BBoxVisualization(cls_dict)				       # 시각화 함수 불러오기
    trt_yolo = TrtYOLO(args.model, args.category_num,
                       args.letter_box)  # 모델 및 파라미터 불러오기

    loop_and_detect(cap, trt_yolo, conf_th=0.3,
                    vis=vis, writer=writer)  # 탐지 시작

    writer.release()
    cap.release()


if __name__ == '__main__':
    main()

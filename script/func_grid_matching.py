# %% package import
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

from load_json import log_read

from func_init_grid import GridArray

# %% Functions

# def object_tracking (pre_info, cur_info, cum_traj) :

# def grid_matching (grid_map, obj_info):

#


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

            # # get_grid
            # grid_matching_time = time.time()
            # grid_coord = ga.update([ref_object_info], ['person'])
            # print("Grid matching time ",
            #       time.time() - grid_matching_time)

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
                print("test")
                print(object_speed)
                object_speed = 0
                print("???")

            speed_set.append(object_speed)
            # print(speed_set)

        obj_behavior_set["speed"] = speed_set

    return obj_behavior_set


# %% Grid initialization
seg_color = [[0, 0, 0], [6, 154, 78], [128, 128, 128], [108, 225, 194]]
seg_img = cv2.imread('../src/seg_map_v2_short.png')

k = 15
shape = (745, 800, 3)

ga = GridArray(k, shape, seg_img)

# %% log data loading and extract objects' infromation by frame

FPS = 10
SAMPLING_RATE = 10
K = 10
P = 10


log_dir = 'logs_by_frame'

dat = log_read(log_dir)

len_dat = len(dat)

person_ref_frame_axies = []
person_ref_frame_label = []

car_ref_frame_axies = []
car_ref_frame_label = []

person_label_cnt = 1
car_label_cnt = 1
min_distance = 30

for frame_index in range(0, len_dat):

    tracking_matching_time = time.time()

    person_cur_frame_axies = []
    person_cur_frame_label = []

    car_cur_frame_axies = []
    car_cur_frame_label = []

    target_frame = dat[frame_index]
    target_frame_info = target_frame[str(frame_index+1)]  # outputs

    for output in target_frame_info:

        left = int(output['min_y'])
        top = int(output['min_x'])
        right = int(output['max_y'])
        bottom = int(output['max_x'])

        clss = output['class']

        lbl = float('nan')

        if clss == 1:  # person

            x = int((top+left) / 2.0)
            y = int(right)

            if (len(person_ref_frame_label) > 0):
                b = np.array([(x, y)])
                a = np.array(person_ref_frame_axies)

                distance = norm(a-b, axis=1)

                car_min_value = distance.min()

                if car_min_value < min_distance:
                    idx = np.where(distance == car_min_value)[0][0]
                    lbl = person_ref_frame_label[idx]

            if math.isnan(lbl):
                lbl = person_label_cnt
                person_label_cnt += 1

            person_cur_frame_label.append(lbl)
            person_cur_frame_axies.append((x, y))

        else:  # no person

            x = int(left)
            y = int(right)

            if (len(car_ref_frame_label) > 0):
                b = np.array([(x, y)])
                a = np.array(car_ref_frame_axies)

                distance = norm(a-b, axis=1)

                person_min_value = distance.min()

                if person_min_value < min_distance:
                    idx = np.where(distance == person_min_value)[0][0]
                    lbl = car_ref_frame_label[idx]

            if math.isnan(lbl):
                lbl = car_label_cnt
                car_label_cnt += 1

            car_cur_frame_label.append(lbl)
            car_cur_frame_axies.append((x, y))

    # HERE: person_ref_frame_label vs person_cur_frame_label
    # XX_ref_frame_label: previous frame information
    # XX_cur_frame_label: current frame information

    # ID 부여 (?), 속도, 방향 추출
    person_obj_behavior = get_object_behaivor(person_ref_frame_label, person_ref_frame_axies,
                                              person_cur_frame_label, person_cur_frame_axies)

    car_obj_behavior = get_object_behaivor(car_ref_frame_label, car_ref_frame_axies,
                                           car_cur_frame_label, car_cur_frame_axies)

    labels = len(person_ref_frame_axies) * ['person']
    person_ref_frame_grid = ga.update(person_ref_frame_axies, labels)

    labels = len(car_ref_frame_axies) * ['car']
    car_ref_frame_grid = ga.update(car_ref_frame_axies, labels)

    if frame_index == 10:
        break

    person_ref_frame_label = person_cur_frame_label
    person_ref_frame_axies = person_cur_frame_axies

    car_ref_frame_label = car_cur_frame_label
    car_ref_frame_axies = car_cur_frame_axies

    print("Matching time for 1 frame: ", time.time() - tracking_matching_time)

    # print("person_ref_frame_label")
    # print(person_ref_frame_label)
    # print("person_ref_frame_axies")
    # print(person_ref_frame_axies)

    # print("car_ref_frame_label")
    # print(car_ref_frame_label)
    # print("car_ref_frame_axies")
    # print(car_ref_frame_axies)


# %%

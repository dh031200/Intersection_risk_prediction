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

from load_json import log_read

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

    ref_len = len(person_ref_frame_label)
    ref_cur = len(person_cur_frame_label)

    if ref_len != 0:

        for ref_trace_index in range(0, ref_len):
            ref_object_id = person_ref_frame_label[ref_trace_index]

            for cur_trace_index in range(0, ref_cur):
                cur_object_id = person_cur_frame_label[cur_trace_index]

                if ref_object_id == cur_object_id:
                    # print("ref_object_id", ref_object_id)
                    # print("cur_object_id", cur_object_id)
                    # print("ref_trace_index", ref_trace_index)
                    # print("cur_trace_index", cur_trace_index)
                    # print("===================")

                    ref_object_info = person_ref_frame_axies[ref_trace_index]
                    cur_object_info = person_cur_frame_axies[cur_trace_index]

                    # print("ref_object_info", ref_object_info)
                    # print("cur_object_info", cur_object_info)

                    # get_speed
                    object_speed = get_speed(ref_object_info, cur_object_info)
                    print(object_speed)

                    # get_heading

                    # get_grid

                    # set_traced_object_info

                    #target_object_info = person_ref_frame_axies[target_object_index]

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

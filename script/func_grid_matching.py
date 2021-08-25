# %% package import
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

# %% log data loading and extract objects' infromation by frame

log_dir = 'logs_by_frame'

dat = log_read(log_dir)

len_dat = len(dat)

ref_frame_axies = []
ref_frame_label = []
label_cnt = 1
min_distance = 10

for frame_index in range(0, len_dat):

    cur_frame_axies = []
    cur_frame_label = []

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

            #print("Come in")

            x = int((top+left) / 2.0)
            y = int(right)

            if (len(ref_frame_label) > 0):
                b = np.array([(x, y)])
                a = np.array(ref_frame_axies)

                distance = norm(a-b, axis=1)

                min_value = distance.min()

                if min_value < min_distance:
                    idx = np.where(distance == min_value)[0][0]
                    lbl = ref_frame_label[idx]

            if math.isnan(lbl):
                lbl = label_cnt
                label_cnt += 1

            cur_frame_label.append(lbl)
            cur_frame_axies.append((x, y))
        # break
    print("ref_frame_label")
    print(ref_frame_label)
    print("ref_frame_axies")
    print(ref_frame_axies)
    ref_frame_label = cur_frame_label
    ref_frame_axies = cur_frame_axies

    if frame_index == 3:
        break


# %%

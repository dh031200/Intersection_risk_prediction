#!/usr/bin/env python
# coding: utf-8

# In[5]:


# %% package import
from typing import final
import numpy as np
import pandas as pd
import math
import time
import json
import ast
import os
import random
from os import listdir
from os.path import isfile, join
from natsort import natsorted
from numpy.linalg import norm
import cv2


# In[15]:


def get_config():
    conf_file = 'config.json'
    with open(conf_file) as json_file:
        config_rs = json.load(json_file)
        
    return config_rs    


# In[19]:


config_ = get_config()
SAMPLING_RATE = int(config_["EXEC_CONFIG"]["SAMPLING_RATE"])
TIME_FLAG_LEG = int(config_["EXEC_CONFIG"]["TIME_FLAG_LEG"])    # unit: sec
DIFF_FRAME = int(config_["EXEC_CONFIG"]["DIFF_FRAME"])
K = int(config_["EXEC_CONFIG"]["K"]) 
P = int(config_["EXEC_CONFIG"]["P"]) 


# In[3]:


def get_all_files(path) :
    
    file_name_list = []
    full_path_list = []
  
    for path, subdirs, files in os.walk(path):
        for name in files:
            file_name_list.append(name)
            full_path_list.append(os.path.join(path, name))
            
    return natsorted(file_name_list), natsorted(full_path_list)

def video_outputs_to_json(outputs, frame_num):

    box_info = outputs['instances'].pred_boxes.tensor.cpu().numpy()
    score = outputs['instances'].scores.cpu().numpy()
    clss = outputs['instances'].pred_classes.cpu().numpy()
    
    target_frame_info = dict()
    
    target_frame_info[str(frame_num)] = []
    
    for obj_idx, obj_info in enumerate(zip(box_info, score, clss)):
        
        bbox_ = obj_info[0]
        score_ = obj_info[1]
        clss_= obj_info[2]
        
        if score_ > 0.8:       
            tmp_dict = dict()
            
            tmp_dict["tmp_id"] = obj_idx
            tmp_dict["min_x"] = bbox_[0]
            tmp_dict["min_y"] = bbox_[1]
            tmp_dict["max_x"] = bbox_[2]
            tmp_dict["max_y"] = bbox_[3]
            tmp_dict["class"] = clss_
            
            target_frame_info[str(frame_num)].append(tmp_dict)

    return target_frame_info


# In[16]:


# %% Functions

def get_transform_matrix(ORIGIN_IMG_COORD, VIRTUAL_IMG_COORD) :
    transform_matrix = cv2.getPerspectiveTransform(np.array(ORIGIN_IMG_COORD, dtype = 'float32'), 
                                               np.array(VIRTUAL_IMG_COORD, dtype = 'float32'))
    
    return transform_matrix
    
def perspective_transform(transform_matrix, target_point_x, target_point_y) : 
    transformed_point = cv2.perspectiveTransform(np.array([[np.array((target_point_x, target_point_y), dtype = 'float32'), ]]), transform_matrix)
    transformed_point = (int(transformed_point[0][0][0]), int(transformed_point[0][0][1]))
    
    return transformed_point[0], transformed_point[1]

def point_in_rect(point, rect):
    x1, y1, x2, y2 = rect
    #x2, y2 = x1 + w, y1+h
    x, y = point
    
    if (x1 < x and x < x2):
        if (y1 < y and y < y2):
            return True        
    return False

def car_tracking(obj_info, car_ref_frame_label, car_ref_frame_axies, car_label_cnt):

    global car_min_distance

    x = int(obj_info[0])
    y = int(obj_info[1])
    lbl = float(obj_info[2])

    b = np.array([(x, y)])
    a = np.array(car_ref_frame_axies)

    distance = norm(a-b, axis=1)

    dist_min_value = distance.min()

    if dist_min_value < car_min_distance:
        idx = np.where(distance == dist_min_value)[0][0]
        lbl = car_ref_frame_label[idx]

    if math.isnan(lbl):
        lbl = car_label_cnt
        car_label_cnt += 1

    return car_ref_frame_label, car_ref_frame_axies, lbl, car_label_cnt


def ped_tracking(obj_info, person_ref_frame_label, person_ref_frame_axies, person_label_cnt):

    global ped_min_distance

    x = int(obj_info[0])
    y = int(obj_info[1])
    lbl = float(obj_info[2])

    b = np.array([(x, y)])
    a = np.array(person_ref_frame_axies)

    distance = norm(a-b, axis=1)

    dist_min_value = distance.min()

    if dist_min_value < ped_min_distance:
        idx = np.where(distance == dist_min_value)[0][0]
        lbl = person_ref_frame_label[idx]

    if math.isnan(lbl):
        lbl = person_label_cnt
        person_label_cnt += 1

    return person_ref_frame_label, person_ref_frame_axies, lbl, person_label_cnt

def get_pixel_distance(x1, y1, x2, y2):

    return math.sqrt((x1-y1)**2 + (x2-y2)**2)


def get_speed(ref_object_info, cur_object_info, FPS):

    # ref_object_info : (x, y)
    # cur_object_info : (x, y)


    pixel_dist = get_pixel_distance(
        ref_object_info[0], ref_object_info[1], cur_object_info[0], cur_object_info[1])

    # calculating with FPS, SAIMPLEING_RATE, and K

    return pixel_dist / FPS

def get_object_behavior(obj_ref_frame_label, obj_ref_frame_axies, obj_cur_frame_label, obj_cur_frame_axies, FPS):

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
                        ref_object_info, cur_object_info, FPS)

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
"""     rs_obj_traj_dict = dict()
    rs_obj_grid_traj_dict = dict()
    rs_obj_bhvr_dict = dict()
    rs_frame_index_dict = dict() """
def frame_filtering(frame_index, obj_traj_dict, obj_grid_traj_dict, obj_bhvr_dict, frame_index_dict):



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


# In[21]:


def random_color():
    levels = range(32,256,32)
    return tuple(random.choice(levels) for _ in range(3))

def draw_traj(ped_traj, car_traj):
        
    img = np.zeros((500, 500, 3), np.uint8)
        
    for obj_idx in ped_traj:
        
        #CLR = COLOR
        target_obj = ped_traj[obj_idx]
        #print(traj[obj_idx])
        for f_idx in range(0, len(target_obj)):
            cv2.circle(img, target_obj[f_idx], 7, (0, 0, 255), -1)
            
    for obj_idx in car_traj:
        
        #CLR = COLOR
        target_obj = car_traj[obj_idx]
        #print(traj[obj_idx])
        for f_idx in range(0, len(target_obj)):
            cv2.circle(img, target_obj[f_idx], 7, (255, 0, 0), -1)
        
    return img


# In[22]:


def draw_frame_object(img, point_list, cls_):
    #img = np.zeros((1000, 1000, 3), np.uint8)
    
    if cls_ == "person":
        CLR = (0, 0, 255)
        
    else: 
        CLR = (255, 0, 0)
        
    for obj_pnt in point_list:
        
        cv2.circle(img, obj_pnt, 7, CLR, -1)
        
    return img
    
    


# In[ ]:


#def draw_roi_to_all_region(cx1, cy1, cx2, cy2, virtual_img_size):
    
    


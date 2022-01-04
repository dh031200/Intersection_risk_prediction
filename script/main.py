#!/usr/bin/env python
# coding: utf-8

# # Packge Import

# In[503]:


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
import natsort
from numpy.linalg import norm
import cv2

from load_json import log_read

import func_init_grid
import importlib
importlib.reload(func_init_grid)

from func_init_grid import setGrid

import utils
importlib.reload(utils)
config_ = utils.get_config()


# # Object detection (Detectron2 Now)

# In[504]:


# Detectron2 ==> Yolo v4
from detectron2.structures import BoxMode
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import ColorMode

from matplotlib import pyplot as plt
from PIL import Image
import ipywidgets as widgets
import IPython.display as display

import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
setup_logger()

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)


# In[505]:


VIDEO_PLAY_FLAG = True

#video_dir = '../NAVIBOX_sample_video/'
video_dir = config_["EXEC_CONFIG"]["WORKSPACE"]
video_files, video_full_path = utils.get_all_files(video_dir)


# In[506]:


ORIGIN_IMG_COORD = [
                    [config_["SITE_CONFIG"]["ORIGIN_IMG_COODR_1"]["X"], config_["SITE_CONFIG"]["ORIGIN_IMG_COODR_1"]["Y"]],
                    [config_["SITE_CONFIG"]["ORIGIN_IMG_COODR_2"]["X"], config_["SITE_CONFIG"]["ORIGIN_IMG_COODR_2"]["Y"]],
                    [config_["SITE_CONFIG"]["ORIGIN_IMG_COODR_3"]["X"], config_["SITE_CONFIG"]["ORIGIN_IMG_COODR_3"]["Y"]],
                    [config_["SITE_CONFIG"]["ORIGIN_IMG_COODR_4"]["X"], config_["SITE_CONFIG"]["ORIGIN_IMG_COODR_4"]["Y"]],
                   ]

VIRTUAL_IMG_COORD = [
                     [config_["SITE_CONFIG"]["VIRTUAL_IMG_COODR"]["MIN"], config_["SITE_CONFIG"]["VIRTUAL_IMG_COODR"]["MIN"]],
                     [config_["SITE_CONFIG"]["VIRTUAL_IMG_COODR"]["MAX"], config_["SITE_CONFIG"]["VIRTUAL_IMG_COODR"]["MIN"]],
                     [config_["SITE_CONFIG"]["VIRTUAL_IMG_COODR"]["MAX"], config_["SITE_CONFIG"]["VIRTUAL_IMG_COODR"]["MAX"]],
                     [config_["SITE_CONFIG"]["VIRTUAL_IMG_COODR"]["MIN"], config_["SITE_CONFIG"]["VIRTUAL_IMG_COODR"]["MAX"]]
                    ]

CRWK_ROI = [
            config_["SITE_CONFIG"]["CRWK_ROI"]["X1"], config_["SITE_CONFIG"]["CRWK_ROI"]["Y1"],
            config_["SITE_CONFIG"]["CRWK_ROI"]["X2"], config_["SITE_CONFIG"]["CRWK_ROI"]["Y2"]
           ]

VIRTUAL_IMG_SIZE = config_["SITE_CONFIG"]["VIRTUAL_IMG_SIZE"]
grid = setGrid(config_["SITE_CONFIG"]["GRID_CELL"], [VIRTUAL_IMG_SIZE, VIRTUAL_IMG_SIZE])

SAMPLING_RATE = int(config_["EXEC_CONFIG"]["SAMPLING_RATE"])

ped_min_distance = int(config_["EXEC_CONFIG"]["PED_MIN_DISTANCE"])
car_min_distance = int(config_["EXEC_CONFIG"]["CAR_MIN_DISTANCE"])
    
SetMode_01 = int(config_["SIGNAL_CONFIG"]["SetMode_01"])
RiskLevel_01 = int(config_["SIGNAL_CONFIG"]["RiskLevel_01"])
RiskLevel_02 = int(config_["SIGNAL_CONFIG"]["RiskLevel_02"])
RiskLevel_03 = int(config_["SIGNAL_CONFIG"]["RiskLevel_03"])
RiskLevel_04 = int(config_["SIGNAL_CONFIG"]["RiskLevel_04"])
RiskLevel_05 = int(config_["SIGNAL_CONFIG"]["RiskLevel_05"])
RiskLevel_06 = int(config_["SIGNAL_CONFIG"]["RiskLevel_06"])

# Perspective transform
transform_matrix = utils.get_transform_matrix(ORIGIN_IMG_COORD, VIRTUAL_IMG_COORD)
#transform_matrix = utils.get_transform_matrix(VIRTUAL_IMG_COORD, ORIGIN_IMG_COORD)


# In[508]:


DISPLAY_FLAG = True
for video_file in video_full_path:
    
    # 여기서 부터 시작
    if DISPLAY_FLAG:
        wImg = widgets.Image(layout = widgets.Layout(border = "solid"))
        display.display(wImg)
        
    cap = cv2.VideoCapture(video_file)        
    
    person_ref_frame_axies = []
    person_ref_frame_label = []
    car_ref_frame_axies = []
    car_ref_frame_label = []
    
    person_label_cnt = 1
    car_label_cnt = 1
    
    lbl_cnt = 1
    
    car_traj_dict = dict()
    ped_traj_dict = dict()
    car_grid_traj_dict = dict()
    ped_grid_traj_dict = dict()

    car_bhvr_dict = dict()
    ped_bhvr_dict = dict()

    car_frame_index_dict = dict()
    ped_frame_index_dict = dict()

    leg_flag = False
    leg_count = 0
    leg_start = time.time()
    
    start_flag = True
    
    frame_num = 0

    #(roi_x1, roi_y1) = utils.perspective_transform(transform_matrix, CRWK_ROI[0], CRWK_ROI[1])
    #(roi_x2, roi_y2) = utils.perspective_transform(transform_matrix, CRWK_ROI[2], CRWK_ROI[3])
    
    #plane_matchin_matrix = plane_refinement(roi_x1, roi_y1, roi_x2, roi_y2, VIRTUAL_IMG_SIZE)
    
    while cap.isOpened():    
        
        if start_flag:
            start_flag = False
            FPS = cap.get(cv2.CAP_PROP_FPS)
            WIDTH = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            HEIGHT = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print("======= Video Info =======")
            print("FPS:", FPS)
            print("Width:", int(WIDTH))
            print("Height:", int(HEIGHT))
            print("==========================")
    
        ret, frame = cap.read()
        
        if frame is None:
            """ EOF """
            break
        
        """ Object detection using Detectron2 Platform ==> YOLOv4 (Bluesignal)"""
        
        frame_num += 1
        if frame_num % SAMPLING_RATE == 0:       
            
            #frame_img = img = np.zeros((VIRTUAL_IMG_SIZE, VIRTUAL_IMG_SIZE, 3), np.uint8) 
            warp_img = cv2.warpPerspective(frame, transform_matrix, (500, 500))
            person_cur_frame_axies = []
            person_cur_frame_label = []
            car_cur_frame_axies = []
            car_cur_frame_label = []
            outputs = predictor(frame)
            
            #if DISPLAY_FLAG:                
                #v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)
                #out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                #drawn_frame = out.get_image()[:, :, ::-1]

            obj_info_in_one_frame = utils.video_outputs_to_json(outputs, frame_num)        
            obj_info_in_one_frame = list(obj_info_in_one_frame.values())[0]

            for obj in obj_info_in_one_frame:

                """Module A: Object Info. Parsing"""            
                left_x = int(obj['min_x'])
                left_y = int(obj['min_y'])
                right_x = int(obj['max_x'])
                right_y = int(obj['max_y'])

                clss = obj['class']
                
                ped_lbl = float('nan')
                car_lbl = float('nan')
                
                """Module B: Object Tracking"""
                if clss == 0: # person

                    x_ = int((left_x+right_x) / 2.0)
                    y_ = int(right_y)
                    
                    if not utils.point_in_rect((x_, y_), CRWK_ROI):
                        continue
                    
                    if DISPLAY_FLAG:
                        cv2.circle(frame, (x_, y_), 3, (255, 0, 0), -1)
                    
                    (x, y) = utils.perspective_transform(transform_matrix, x_, y_)
                    #(x, y) = plane_rematching(x, y, plane_matchin_matrix)

                    if len(person_ref_frame_label) > 0:

                        b = np.array([(x, y)])
                        a = np.array(person_ref_frame_axies)

                        ped_distance = norm(a-b, axis = 1)

                        ped_dist_min_value = ped_distance.min()

                        if ped_dist_min_value < ped_min_distance:
                            ped_idx = np.where(ped_distance == ped_dist_min_value)[0][0]
                            ped_lbl = person_ref_frame_label[ped_idx]

                        if math.isnan(ped_lbl):
                            ped_lbl = person_label_cnt
                            person_label_cnt += 1

                    else:
                        person_ref_frame_label.append(ped_lbl)
                        person_ref_frame_axies.append((x, y))

                    person_cur_frame_label.append(ped_lbl)
                    person_cur_frame_axies.append((x, y))
                    
                    if DISPLAY_FLAG:
                        warp_img = utils.draw_frame_object(warp_img, person_ref_frame_axies, "person")
                    
                    

                elif clss == 2: # add other class following if-elif-else phras if need other vehicle types

                    x_ = int((left_x+right_x) / 2.0)
                    y_ = int(right_y)
                        
                    if not utils.point_in_rect((x_, y_), CRWK_ROI):
                        continue
                    
                    if DISPLAY_FLAG:
                        cv2.circle(frame, (x_, y_), 3, (255, 255, 0), -1)
                        
                    (x, y) = utils.perspective_transform(transform_matrix, x_, y_)                    
                    #(x, y) = plane_rematching(x, y, plane_matchin_matrix)

                    if len(car_ref_frame_label) > 0:
                        b = np.array([(x, y)])
                        a = np.array(car_ref_frame_axies)

                        car_distance = norm(a-b, axis = 1)

                        car_dist_min_value = car_distance.min()

                        if car_dist_min_value < car_min_distance:
                            car_idx = np.where(car_distance == car_dist_min_value)[0][0]
                            car_lbl = car_ref_frame_label[car_idx]

                        if math.isnan(car_lbl):
                            car_lbl = car_label_cnt
                            car_label_cnt += 1

                    else:
                        car_ref_frame_label.append(car_lbl)
                        car_ref_frame_axies.append((x, y))

                    car_cur_frame_label.append(car_lbl)
                    car_cur_frame_axies.append((x, y))
                    
                    if DISPLAY_FLAG:
                        warp_img = utils.draw_frame_object(warp_img, car_ref_frame_axies, "car")
                    

                else:
                    continue
                
#             print("person_ref_frame_label")
#             print(person_ref_frame_label)
#             print("person_cur_frame_axies")
#             print(person_cur_frame_axies)
#             print("car_ref_frame_label")
#             print(car_ref_frame_label)
#             print("car_ref_frame_axies")
#             print(car_ref_frame_axies)

            """Module C: Trajectory Parsing and Behavior Info. Extracting with Grid Matching"""
            ped_grid_list = grid.get_grid_from_coord(person_ref_frame_axies, ['1']*len(person_ref_frame_axies))
            car_grid_list = grid.get_grid_from_coord(car_ref_frame_axies, ['1']*len(car_ref_frame_axies))

            ped_bhvr_list = utils.get_object_behavior(person_ref_frame_label, person_ref_frame_axies, person_cur_frame_label, person_cur_frame_axies, FPS)
            car_bhvr_list = utils.get_object_behavior(car_ref_frame_label, car_ref_frame_axies, car_cur_frame_label, car_cur_frame_axies, FPS)        

            ped_traj_dict, ped_grid_traj_dict, ped_bhvr_dict, ped_frame_index_dict = utils.get_obj_info(person_ref_frame_label, person_ref_frame_axies,
                                                                                                        ped_grid_list, ped_bhvr_list,
                                                                                                        ped_traj_dict, ped_grid_traj_dict, ped_bhvr_dict,
                                                                                                        frame_num, ped_frame_index_dict)

            car_traj_dict, car_grid_traj_dict, car_bhvr_dict, car_frame_index_dict = utils.get_obj_info(car_ref_frame_label, car_ref_frame_axies,
                                                                                                        car_grid_list, car_bhvr_list,
                                                                                                        car_traj_dict, car_grid_traj_dict, car_bhvr_dict,
                                                                                                        frame_num, car_frame_index_dict)
            
            
            
            # 분석 대상 trajectory
            ped_traj_dict, ped_grid_traj_dict, ped_bhvr_dict, ped_frame_index_dict = utils.frame_filtering(frame_num,
                                                                                                           ped_traj_dict,
                                                                                                           ped_grid_traj_dict,
                                                                                                           ped_bhvr_dict,
                                                                                                           ped_frame_index_dict)

            car_traj_dict, car_grid_traj_dict, car_bhvr_dict, car_frame_index_dict = utils.frame_filtering(frame_num,
                                                                                                           car_traj_dict,
                                                                                                           car_grid_traj_dict,
                                                                                                           car_bhvr_dict,
                                                                                                           car_frame_index_dict)
            
            

            ## prediction
            
            
            ## 알람 (대상 object ID 기억해야함)
            ## 한번 경고 한 녀석은 더이상 취급하지 않음            
            
            
            if DISPLAY_FLAG:
                img = utils.draw_traj(ped_traj_dict, car_traj_dict)
                #cv2.rectangle(img, ((roi_x1, roi_y1), (roi_x2, roi_y2), (0, 0, 255), 3)
   
                #img = cv2.resize(img, (int(WIDTH), int(HEIGHT)))
                img = cv2.resize(img, (500, 500))
                
                #drawn_frame = cv2.resize(drawn_frame, (500, 500))
                #tt_frame = frame.copy()
                #tt_frame = cv2.resize(tt_frame, (500, 500))
                # ROI Draw
                cv2.rectangle(frame, (CRWK_ROI[0], CRWK_ROI[1]), (CRWK_ROI[2], CRWK_ROI[3]), (0, 0, 255), 3)
                tt_frame = frame.copy()
                tt_frame = cv2.resize(tt_frame, (500, 500))
                
                rs_img = cv2.hconcat([tt_frame, warp_img, img])
                tmpStream = cv2.imencode(".jpeg", rs_img)[1].tobytes()
                wImg.value = tmpStream      
                
            person_ref_frame_label = person_cur_frame_label
            person_ref_frame_axies = person_cur_frame_axies

            car_ref_frame_label = car_cur_frame_label
            car_ref_frame_axies = car_cur_frame_axies
            
            #if frame_num > 20:                
            #    break # one frame break
    
            #print("ped_traj_dict")
            #print(ped_traj_dict)
            #print("car_traj_dict")
            #print(car_traj_dict)
            
#     img = draw_traj(car_traj_dict, "car")
#     plt.figure(figsize=(10, 10))
#     plt.imshow(img)    
    
#     img = draw_traj(ped_traj_dict, "person")
#     plt.figure(figsize=(10, 10))
#     plt.imshow(img)           

    break # one video file break        


# In[501]:


transform_matrix = utils.get_transform_matrix(ORIGIN_IMG_COORD, VIRTUAL_IMG_COORD)
#transform_matrix = utils.get_transform_matrix(VIRTUAL_IMG_COORD, ORIGIN_IMG_COORD)
warp_img = cv2.warpPerspective(tt_frame, transform_matrix, (500, 500))
plt.figure(figsize=(10, 10))
plt.imshow(warp_img)

warp_img = cv2.warpPerspective(frame, transform_matrix, (500, 500))


# In[ ]:


img.shape


# In[ ]:


cv2.imwrite("YS_background.jpg", frame)


# In[490]:


def plane_refinement(roi_x1, roi_y1, roi_x2, roi_y2, VIRTUAL_IMG_SIZE):
  
    origin_rect = [[roi_x1, roi_y1],  [roi_x2, roi_y1], [roi_x2, roi_y2], [roi_x1, roi_y2]]
    new_rect = [[0, 0], [VIRTUAL_IMG_SIZE, 0], [VIRTUAL_IMG_SIZE, VIRTUAL_IMG_SIZE], [0, VIRTUAL_IMG_SIZE]]
    transform_matrix = cv2.getPerspectiveTransform(np.array(origin_rect, dtype = 'float32'), 
                                                   np.array(new_rect, dtype = 'float32'))
    
    return transform_matrix
    
def plane_rematching(x, y, transform_matrix):
    
    transformed_point = cv2.perspectiveTransform(np.array([[np.array((x, y), dtype = 'float32'), ]]), transform_matrix)
    transformed_point = (int(transformed_point[0][0][0]), int(transformed_point[0][0][1]))
    
    return transformed_point[0], transformed_point[1]


# In[405]:


VIRTUAL_IMG_SIZE


# In[ ]:





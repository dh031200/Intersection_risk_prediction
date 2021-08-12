"""trt_yolo_cv.py

This script could be used to make object detection video with
TensorRT optimized YOLO engine.

"cv" means "create video"
made by BigJoon (ref. jkjung-avt)
"""

import json
import os
import argparse
import time
import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver

from utils.yolo_classes import get_cls_dict
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO


def parse_args():
    """Parse input arguments."""
    desc = ('Run the TensorRT optimized object detecion model on an input '
            'video and save BBoxed overlaid output as another video.')
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        '-v', '--video', type=str, required=True,
        help='input video file name')
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

def get_object_log_file (frame_num, boxes, clss, confs) : # write the log file (object info) with boxes and classes
    
    log_info_dat = []

    frame_info_dict = dict()

    for object_index in range(0, len(clss)):

        if confs[object_index] > 0.6 :
            tmp_dict = dict()

            tmp_dict = {'tmp_id': int(object_index),
                    'min_x': int(boxes[object_index][0]),
                    'min_y': int(boxes[object_index][1]),
                    'max_x': int(boxes[object_index][2]),
                    'max_y': int(boxes[object_index][3]),
                    'class': int(clss[object_index])}

            log_info_dat.append(tmp_dict)

    frame_info_dict[frame_num] = log_info_dat
    
    log_name = 'logs_by_frame/log_final_{}.json'.format(frame_num)
    f = open(log_name, 'w')

    f.write(json.dumps(frame_info_dict))

    f.close()



def loop_and_detect(cap, trt_yolo, conf_th, vis):
    """Continuously capture images from camera and do object detection.

    # Arguments
      cap: the camera instance (video source).
      trt_yolo: the TRT YOLO object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
      writer: the VideoWriter object for the output video.
    """
    pt = 0
    frame_count = 0 
    while True:
        ret, frame = cap.read()
        if frame is None:  break
        boxes, confs, clss = trt_yolo.detect(frame, conf_th)

#        print("boxes", boxes)
#        print("class", clss)

        frame_count += 1
        get_object_log_file(frame_count, boxes, clss, confs)

        #frame = vis.draw_bboxes(frame, boxes, confs, clss)
        st = time.time()
        sec = st - pt
        pt = st
        fps = 1 / sec
        fstr = 'FPS : %01.f' % fps
        #cv2.putText(frame, fstr, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
        #cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print('\nDone.')


def main():
    args = parse_args()
    if args.category_num <= 0:
        raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
    if not os.path.isfile('yolo/%s.trt' % args.model):
        raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % args.model)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit('ERROR: failed to open the input video file!')
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    cls_dict = get_cls_dict(args.category_num)
    vis = BBoxVisualization(cls_dict)
    trt_yolo = TrtYOLO(args.model, args.category_num, args.letter_box)
    loop_and_detect(cap, trt_yolo, conf_th=0.3, vis=vis)
    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    main()

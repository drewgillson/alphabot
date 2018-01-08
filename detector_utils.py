""" _    _       _           _           _
   / \  | |_ __ | |__   __ _| |__   ___ | |_
  / _ \ | | '_ \| '_ \ / _` | '_ \ / _ \| __|
 / ___ \| | |_) | | | | (_| | |_) | (_) | |_
/_/   \_\_| .__/|_| |_|\__,_|_.__/ \___/ \__|
          |_|
A screen-less interactive spelling primer powered by computer vision

Copyright (C) 2018  Drew Gillson <drew.gillson@gmail.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import numpy as np
import sys
import tensorflow as tf
import os
from threading import Thread
import cv2
import label_map_utils


sys.path.append("..")

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = 'hand_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('hand_label_map.pbtxt')

NUM_CLASSES = 1
# load label map
label_map = label_map_utils.load_labelmap(PATH_TO_LABELS)
categories = label_map_utils.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_utils.create_category_index(categories)

def load_inference_graph():
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)
    return detection_graph, sess


def get_box_coords(boxes, i, cap_params):
    return (int(boxes[i][1] * cap_params['im_width']), int(boxes[i][3] * cap_params['im_width']),
            int(boxes[i][0] * cap_params['im_height']), int(boxes[i][2] * cap_params['im_height']))

def get_touched_letter(cap_params, scores, boxes, image_np, args):
    import scipy.misc

    cp = image_np.copy()
    for i in range(cap_params['num_hands_detect']):

        if (scores[i] > cap_params['hand_score_thresh']):
            left, right, top, bottom = get_box_coords(boxes, i, cap_params)

            squares = []
            crops = []

            squares.append((left-10, left+90, top-90, top+10))
            #squares.append((left-10, left+70, top-70, top+10))
            #squares.append((left+10, left+90, top-70, top+10))
            #squares.append((left-20, left+60, top-50, top+30))
            #squares.append((left+20, left+100, top-50, top+30))

            for idx, square in enumerate(squares):
                left, right, top, bottom = square

                crop_img = cp[top:bottom, left:right]
                if crop_img.shape[0] == crop_img.shape[1]:
                    crop_img = cv2.fastNlMeansDenoisingColored(crop_img, None, 4, 4, 7, 21)
                    crop_img = cv2.Canny(crop_img, 90, 100)
                    crop_img = scipy.misc.imresize(crop_img, (28, 28))
                    if args.debug:
                        cv2.rectangle(image_np, (left, top), (right, bottom), (77, 255, 9), 2, 1)
                        cv2.imshow('crop' + str(idx), crop_img)
                    crops.append(crop_img)

            return crops


def detect_objects(image_np, detection_graph, sess):
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name(
        'detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name(
        'detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name(
        'detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name(
        'num_detections:0')

    image_np_expanded = np.expand_dims(image_np, axis=0)

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores,
            detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    return np.squeeze(boxes), np.squeeze(scores)


# Code to thread reading camera input.
# Source : Adrian Rosebrock
# https://www.pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/
class WebcamVideoStream:
    def __init__(self, src, width, height):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.frame

    def size(self):
        # return size of the capture device
        return self.stream.get(3), self.stream.get(4)

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

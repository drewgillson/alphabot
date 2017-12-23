import numpy as np
import sys
import tensorflow as tf
import os
from threading import Thread
import cv2
import label_map_utils


sys.path.append("..")

# score threshold for showing bounding boxes.
_score_thresh = 0.40

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


def get_touched_letter(num_hands_detect, score_thresh, scores, boxes, im_width, im_height, image_np):
    import scipy.misc

    cp = image_np.copy()
    for i in range(num_hands_detect):
        if (scores[i] > score_thresh):
            (left, right, top, bottom) = (int(boxes[i][1] * im_width), int(boxes[i][3] * im_width),
                                          int(boxes[i][0] * im_height), int(boxes[i][2] * im_height))

            cv2.rectangle(image_np, (left, top), (right, bottom), (77, 255, 9), 1, 1)

            squares = []
            crops = []

            squares.append((left, left+80, top-70, top+10))
            #squares.append((x-20, x+60, y-90, y-10))
            #squares.append((x+20, x+100, y-90, y-10))
            #squares.append((x-20, x+60, y-50, y+30))
            #squares.append((x+20, x+100, y-50, y+30))

            for idx, square in enumerate(squares):
                left, right, top, bottom = square
                cv2.rectangle(image_np, (left, top), (right, bottom), (77, 255, 9), 1, 1)
                crop_img = cp[top:bottom, left:right]
                if crop_img.shape[0] == crop_img.shape[1]:
                    crop_img = cv2.fastNlMeansDenoisingColored(crop_img, None, 4, 4, 7, 21)
                    crop_img = cv2.Canny(crop_img, 90, 100)
                    crop_img = scipy.misc.imresize(crop_img, (28, 28))
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

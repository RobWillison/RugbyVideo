import cv2
import numpy as np
import math
import random
import imutils
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../Mask_RCNN")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

def find_people(frame):
    # Run detection
    results = model.detect([frame], verbose=1)
    print(results)
    results = zip(results[0]['rois'], results[0]['class_ids'])
    players = []
    for box, class_id in results:
        if class_id != 1:
            continue
        cv2.rectangle(frame, (box[1], box[0]), (box[3], box[2]), (0,0,255), 2)
        players.append((box[1], box[0], box[3], box[2]))

    return frame, players



def start_track(frame, box):
    bbox = (box[0], box[1], box[2] - box[0], box[3] - box[1])
    tracker = cv2.TrackerCSRT_create()
    ok = tracker.init(frame, bbox)
    return tracker

def track_box(tracker, frame):
    _, box = tracker.update(frame)
    p1 = (int(box[0]), int(box[1]))
    p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
    cv2.rectangle(frame, p1, p2, (0,255,0), 2)
    return box

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('video.mp4')

# Check if camera opened successfully
if (cap.isOpened()== False):
  print("Error opening video stream or file")

cap.set(0,60720);
trackers = []
frame_count = 0
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()

  if frame_count % 100:
      cv2.imwrite('img/' + str(frame_count) + '.jpg', frame)
  # if ret == True:
      # for i in range(len(trackers)):
      #     box = track_box(trackers[i], frame)
      #
      # if frame_count % 5 == 0:
      #     trackers = []
      #     frame, players = find_people(frame)
      #     for player in players:
      #         trackers.append(start_track(frame, player))
      #
      # cv2.imshow('Frame',frame)
      # cv2.waitKey(0)


  frame_count += 1

cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()

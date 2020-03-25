import cv2
import numpy as np
import math
import random
from pitch import Pitch



# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('data/video.mp4')

# Check if camera opened successfully
if (cap.isOpened()== False):
  print("Error opening video stream or file")

cap.set(0,62200);

pitch = Pitch(['BL', 'I', 'I', 'I', 'TL', 'SL', '20M'])
frame_count = 0
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()

  if ret == True:
    if frame_count % 5 == 0:
        pitch.update(frame, True)
    else:
        pitch.update(frame)

    pitch.annotate(frame)

    cv2.imshow('Frame',frame)
    cv2.waitKey(0)
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

  # Break the loop
  else:
    break

  frame_count += 1

cv2.waitKey(0)
# When everything done, release the video capture object
cap.release()
cv2.destroyAllWindows()

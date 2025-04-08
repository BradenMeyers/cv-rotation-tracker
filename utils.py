import cv2
import numpy as np
from config import *

# Function to detect the orange object
def detect_orange_object(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=erode_iter)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask

def find_centroid(mask):
    if np.count_nonzero(mask) == 0:
        return None
    moments = cv2.moments(mask)
    if moments["m00"] != 0:
        return (int(moments["m10"] / moments["m00"]), 
                int(moments["m01"] / moments["m00"]))
    return None


def crop(frame):
    return frame[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

def crop_circle(frame):
    frame = crop(frame)
    # Create a circular mask
    height, width, _ = frame.shape
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, pixel_center, radius, 255, -1)

    return cv2.bitwise_and(frame, frame, mask=mask)
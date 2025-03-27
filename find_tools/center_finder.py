import cv2
import numpy as np

from config import *
from utils import *

# Global variables
point = None


def select_point(event, x, y, flags, param):
    global point
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)
        print(f"Point selected at: {point}")


def find_center():
    global point
    point = None
    center = None
    cap = cv2.VideoCapture(video_path)
    cv2.namedWindow('Center Finder')
    cv2.setMouseCallback('Center Finder', select_point)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = crop(frame)
        
        if point:
            center = point
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            cv2.putText(frame, f"Center: {center}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Click to set center", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Resize the frame to fit the screen
        cv2.imshow('Center Finder', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if center:
        print(f"Final center coordinates: {center}")
        print("Use these coordinates in the rotation counter script.")
    else:
        print("No center was set.")

    return center

def find_radius(center):
    global point
    point = None
    radius_point = None
    radius = None
    cap = cv2.VideoCapture(video_path)
    cv2.namedWindow('Radius Finder')
    cv2.setMouseCallback('Radius Finder', select_point)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = crop(frame)
        
        if point:
            radius_point = point
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            # Draw line from center to radius point
            radius = int(np.sqrt((center[0] - radius_point[0])**2 + (center[1] - radius_point[1])**2))
            cv2.circle(frame, center, radius, (0, 255, 0), 2)
            cv2.putText(frame, f"radius: {radius}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Click to set radius", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        
        # Resize the frame to fit the screen
        cv2.imshow('Radius Finder', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if radius_point:
        print(f"Final radius coordinates: {radius_point}")
        # calculate the pixel distance between the center and the radius point
        print(f"Radius: {radius}")
        print("Use these coordinates in the rotation counter script.")
    else:
        print("No radius was set.")

    return int(radius)

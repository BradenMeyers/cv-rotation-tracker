import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

from config import *
from utils import *
from find_tools.center_finder import find_center, find_radius

center  = find_center()

radius = find_radius(center)

# Open the video file
cap = cv2.VideoCapture(video_path)

mask = None

# Process the video frame by frame
for i in range(frames_per_rotation):
    # Read the frame
    ret, frame = cap.read()
    if not ret:
        print(f"Failed to read frame {i}")
        break

    frame = crop(frame)

    if mask is None:
        # Create a circular mask
        height, width, _ = frame.shape
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, -1)

    # Apply the circular mask to the frame
    circular_cropped_frame = cv2.bitwise_and(frame, frame, mask=mask)
    
    # # Crop the frame to the bounding box of the circle
    # x, y = center
    # x1, y1 = max(0, x - radius), max(0, y - radius)
    # x2, y2 = min(width, x + radius), min(height, y + radius)
    # circular_cropped_frame = circular_cropped_frame[y1:y2, x1:x2]


    # Display the circularly cropped frame
    cv2.imshow('Circular Cropped Video', circular_cropped_frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

    if delay_video:
        time.sleep(video_delay)

# Release the video capture object
cap.release()
cv2.destroyAllWindows()

# Print the circular crop parameters for reference
print(f"Circular cropping parameters - Center: ({center}), Radius: {radius}")
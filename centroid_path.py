import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

from config import *
from utils import *

# Initialize video capture
cap = cv2.VideoCapture(video_path)

# Define the point to check crossing
# crossing_point = 900

# Store centroid positions
centroids_x = []
centroids_y = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    cropped_frame = crop_circle(frame)
    
    mask = detect_orange_object(cropped_frame)
    centroid = find_centroid(mask)
    
    # Convert mask to 3 channels to display it properly
    mask_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    cv2.circle(mask_display, center, 5, (0, 0, 255), -1)
    
    # # Draw the crossing line
    # cv2.line(mask_display, (crossing_point, 0), (crossing_point, cropped_frame.shape[0]), (255, 0, 0), 2)
    
    # Draw the centroid if detected
    if centroid:
        cx, cy = centroid
        cv2.circle(mask_display, (cx, cy), 5, (0, 255, 0), -1)
        centroids_x.append(cx)
        centroids_y.append(cy)
    # time.sleep(0.1)
    cv2.imshow('Mask with Line and Centroid', mask_display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Plot centroid path
plt.figure(figsize=(8, 6))
plt.plot(centroids_x, centroids_y, marker="o", linestyle="-", color="orange", label="Centroid Path")

# Plot the center point
plt.plot(center[0], center[1], marker="x", color="red", label="Center Point")
# Add arrows for direction
for i in range(len(centroids_x) - 1):
    plt.arrow(centroids_x[i], centroids_y[i], 
              centroids_x[i+1] - centroids_x[i], 
              centroids_y[i+1] - centroids_y[i], 
              shape='full', lw=0, length_includes_head=True, head_width=5, color='blue')

# # Draw the crossing line
# plt.axvline(x=crossing_point, color='red', linestyle='--', label='Crossing Line')

plt.axis('equal')
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("Centroid Path Over Time")
plt.legend()
plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
plt.show()

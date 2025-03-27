import cv2
import numpy as np
import math
import time
import matplotlib.pyplot as plt

from config import *
from utils import *
import os

# Global variables
prev_angle = None
cumulative_angle = 0.0
counting_started = False

# Rotation tracking variables
rotation_count = 0
frame_count = 0
frames_per_rotation = []
start_time = time.time()
delta_angles = []
angle_list = []
pixel_count_list = []
rotation_pos_list = []


# Initialize video capture
filename = os.path.splitext(os.path.basename(video_path))[0]
output_path = f"output/{filename}_output.mov"
cap = cv2.VideoCapture(video_path)

if display_video:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, 
                     (crop_x_max - crop_x_min, crop_y_max - crop_y_min))


cropped_frame = None
mask = None
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    cropped_frame = crop_circle(frame)
    frame_count += 1

    mask = detect_orange_object(cropped_frame)
    centroid = find_centroid(mask)

    if centroid:
        cx, cy = centroid
        dx = cx - center[0]
        dy = center[1] - cy  # Flip Y-axis for proper angle calculation
        current_angle = math.atan2(dy, dx)

        cv2.line(cropped_frame, center, (cx, cy), (255, 0, 0), 2)
        cv2.circle(cropped_frame, (cx, cy), 5, (0, 255, 0), -1)

        if prev_angle is not None:
            # Calculate angle delta and adjust for wrapping
            delta = current_angle - prev_angle
            if delta > math.pi:
                delta -= 2 * math.pi
            elif delta < -math.pi:
                delta += 2 * math.pi

            cumulative_angle += delta
            angle_list.append(cumulative_angle)
            delta_angles.append(delta)
            pixel_count_list.append(cv2.countNonZero(mask))

            # Check for full rotations
            if abs(cumulative_angle) >= 2 * math.pi:
                rotation_pos_list.append((cx, cy))
                rotation_count += 1
                cumulative_angle = math.copysign(
                    abs(cumulative_angle) % (2 * math.pi), 
                    cumulative_angle
                )

        prev_angle = current_angle

    if display_video:
        # Draw rotation info
        cv2.circle(cropped_frame, center, 5, (0, 0, 255), -1)
        cv2.putText(cropped_frame, f"Rotations: {rotation_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(cropped_frame, f"Current Angle: {math.degrees(current_angle):.2f} degrees",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Find contours of the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours on the blank image
        cv2.drawContours(cropped_frame, contours, -1, (255, 255, 255), 2)  # Green color, 1px thick

        # Write and display frame
        out.write(cropped_frame)
        cv2.imshow('Rotation Counter', cropped_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# Calculate RPM
total_angle = sum(delta_angles)     #angle in radians
total_rotation = total_angle/(2*math.pi)


total_time = frame_count / fps  # time in seconds
total_min = total_time / 60
rpm = total_rotation / total_min

print(f"Total rotations: {total_rotation}")
print(f"Estimated RPM: {rpm}")


# Perform analysis on delta_angles
mean_delta = np.mean(delta_angles)
std_dev_delta = np.std(delta_angles)
max_delta = np.max(delta_angles)
min_delta = np.min(delta_angles)

mean_delta_rev = mean_delta / (math.pi * 2)
std_dev_delta_rev = std_dev_delta / (math.pi * 2)


# Plot histogram of delta angles
plt.hist(delta_angles, bins=30, edgecolor='black')
plt.title('Histogram of Delta Angles')
plt.xlabel('Delta Angle (radians)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

print(f"Mean of delta angles: {mean_delta:.4f} radians, {mean_delta_rev:.4f} revs")
print(f"Standard deviation of delta angles: {std_dev_delta:.4f} radians, {std_dev_delta_rev:.4f} revs")
print(f"Max delta angle: {max_delta:.4f} radians")
print(f"Min delta angle: {min_delta:.4f} radians")
print(f"Total frames: {frame_count}")
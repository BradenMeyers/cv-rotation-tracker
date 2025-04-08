import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

from config import *
from utils import *
import os

# Global variables
prev_angle = None
cumulative_angle = 0.0
counting_started = False
first_frame = None

# Rotation tracking variables
rotation_count = 0      # Number of full rotations
frame_count = 0         # Frame counter
frames_per_rotation = [] # Store frames per rotation
delta_angles = []       # Store delta angles
angle_list = []         # Store angles
pixel_count_list = []   # Store pixel counts
rotation_pos_list = []  # Store rotation positions
centroids_path = []        # Store centroid positions
full_rotations_centroid_locations = []  # Store all centroid locations of full rotations (for actual center calculation)

# Initialize video playback
filename = os.path.splitext(os.path.basename(video_path))[0]
cap = cv2.VideoCapture(video_path)

if display_video:
    output_path = f"output/{filename}_output.mov"
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
    mask = detect_orange_object(cropped_frame)
    centroid = find_centroid(mask)

    if first_frame is None:
        # Set the center of the circle
        first_frame = cropped_frame.copy()

    frame_count += 1

    if centroid:
        cx, cy = centroid
        dx = cx - center[0]
        dy = center[1] - cy  # Flip Y-axis for proper angle calculation
        current_angle = math.atan2(dy, dx)
        centroids_path.append((cx, cy))

        cv2.line(cropped_frame, pixel_center, (cx, cy), (255, 0, 0), 2)
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
                full_rotations_centroid_locations = centroids_path.copy()
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

if full_rotations_centroid_locations:
    # Convert to numpy array for easy mean calculation
    rotation_pos_array = np.array(full_rotations_centroid_locations)
    mean_x = np.mean(rotation_pos_array[:, 0])
    mean_y = np.mean(rotation_pos_array[:, 1])
    estimated_center = (mean_x, mean_y)
    print(f"Estimated Center of Rotation: {estimated_center}")
    print(f"Updating to new center can produce higher accuracy.")
else:
    print("No full rotations detected; cannot estimate center.")


# Perform analysis on delta_angles
mean_delta = np.mean(delta_angles)
std_dev_delta = np.std(delta_angles)
max_delta = np.max(delta_angles)
min_delta = np.min(delta_angles)

mean_delta_rev = mean_delta / (math.pi * 2)
std_dev_delta_rev = std_dev_delta / (math.pi * 2)


###################### PLOTTING ######################
if degrees:
    units = "degrees"
    delta_angles = [math.degrees(angle) for angle in delta_angles]
    angle_list = [math.degrees(angle) for angle in angle_list]
    mean_delta = math.degrees(mean_delta)
    max_delta = math.degrees(max_delta)
    min_delta = math.degrees(min_delta)
    std_dev_delta = math.degrees(std_dev_delta)

print(f"Mean of delta angles: {mean_delta} {units}, {mean_delta_rev} revolutions")
print(f"Standard deviation of delta angles: {std_dev_delta} {units}, {std_dev_delta_rev} revolutions")
print(f"Max delta angle: {max_delta} {units}")
print(f"Min delta angle: {min_delta} {units}")
print(f"Total frames: {frame_count}")

# Plot histogram of delta angles
plt.hist(delta_angles, bins=30, edgecolor='black')
plt.title('Distribution of Delta Angles')
plt.xlabel(f'Delta Angle ({units})')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Histogram of X and Y Positions
plt.figure(figsize=(12, 6))
x_vals, y_vals = zip(*rotation_pos_list) if rotation_pos_list else ([], [])

# X and Y positions vs Time
x_len = range(len(x_vals))  # Assuming each position corresponds to a time step
y_len = range(len(y_vals))  # Assuming each position corresponds to a time step

plt.subplot(1, 3, 1)
plt.plot(x_len, x_vals, marker='o', linestyle='-', alpha=0.7)
plt.xlabel("Rotation Count")
plt.ylabel("X Centroid Position")
plt.title("X Centroid Position vs Rotation")
plt.grid()

plt.subplot(1, 3, 2)
plt.plot(y_len, y_vals, marker='o', linestyle='-', alpha=0.7)
plt.xlabel("Rotation Count")
plt.ylabel("Y Centroid Position")
plt.title("Y Position vs Rotation")
plt.grid()

# Plot the Centroid positions at each rotation on the image
plt.subplot(1, 3, 3)
plt.scatter(x_vals, y_vals, marker='o', color='green', label="Centroid Location")
plt.plot(center[0], center[1], marker="x", color="red", label="Center Point")
# Plot the first frame rgb image
plt.imshow(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
plt.axis('equal')
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("Centroid Location in Frame")
plt.legend()
plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates

plt.tight_layout()
plt.show()

# Delta Angle vs. Pixel Count
plt.subplot(1, 3, 1)
# plt.figure()
plt.scatter(delta_angles, pixel_count_list, alpha=0.7)
plt.xlabel(f"Delta Angle ({units})")
plt.ylabel("Pixel Count")
plt.title("Delta Angle vs. Pixel Count")
plt.grid()
# plt.show()

# Delta Angle vs. Actual Angle Over Time
plt.subplot(1, 3, 2)
# plt.figure()
plt.scatter(delta_angles, angle_list, alpha=0.7)
plt.xlabel(f"Delta Angle ({units})")
plt.ylabel(f"Angular Position ({units})")
plt.title("Angular Position vs Delta Angle")
# plt.legend()
plt.grid()
# plt.show()

# Angular Position vs. Pixel Count 
plt.subplot(1, 3, 3)
# plt.figure()
plt.scatter(pixel_count_list, angle_list, alpha=0.7)
plt.xlabel("Pixel Count")
plt.ylabel(f"Angular Position ({units})")
plt.title("Angular Position vs Pixel Count")
# plt.legend()
plt.grid()
plt.show()


# Centroid path
centroids_x, centroids_y = zip(*centroids_path) if centroids_path else ([], [])
plt.figure(figsize=(8, 6))
plt.plot(centroids_x, centroids_y, marker="o", linestyle="-", color="green", label="Centroid Path")

# Plot the center point
plt.plot(center[0], center[1], marker="x", color="red", label="Center Point")
# Add arrows for direction
for i in range(len(centroids_x) - 1):
    plt.arrow(centroids_x[i], centroids_y[i], 
            centroids_x[i+1] - centroids_x[i], 
            centroids_y[i+1] - centroids_y[i], 
            shape='full', lw=0, length_includes_head=True, head_width=5, color='blue')
plt.imshow(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
plt.axis('equal')
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("Centroid Path Over Time")
plt.legend()
plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
plt.show()
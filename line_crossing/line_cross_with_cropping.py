import cv2
import numpy as np
import time

# Variables to calculate average frames per rotation
frame_count = 0
rotation_count = 0
start_time = time.time()
frames_per_rotation = []
counting_started = False

# Cropping parameters
crop_x_min = 370  # Adjust as needed
crop_y_min = 0  # Adjust as needed
crop_x_max = 1350  # Adjust as needed
crop_y_max = 670  # Adjust as needed

# Function to detect the orange object
def detect_orange_object(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([10, 100, 100])
    upper_orange = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    
    # Erode to remove small noise and then dilate to connect all pixels
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    return mask

# Function to find the centroid of the detected object
def find_centroid(mask):
    if np.count_nonzero(mask) == 0:
        print("No orange object detected in this frame.")
        return None
    
    moments = cv2.moments(mask)
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        return cx, cy
    return None

# Initialize video capture
cap = cv2.VideoCapture('Videos/IMG_0860.mov')

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can use other codecs like 'MJPG', 'X264', etc.
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (crop_x_max - crop_x_min, crop_y_max - crop_y_min))

# Define the point to check crossing
crossing_point = 900
last_crossing = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Crop the frame
    cropped_frame = frame[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
    
    frame_count += 1
    mask = detect_orange_object(cropped_frame)
    centroid = find_centroid(mask)
    
    # Draw the crossing line on the original frame coordinates
    cv2.line(cropped_frame, (crossing_point - crop_x_min, 0), (crossing_point - crop_x_min, cropped_frame.shape[0]), (255, 0, 0), 2)
    
    # Draw the centroid if detected
    if centroid:
        cx, cy = centroid
        cv2.circle(cropped_frame, (cx, cy), 5, (0, 255, 0), -1)
        
        # Check if the centroid crosses the defined point
        if last_crossing is not None and cx < (crossing_point - crop_x_min) and last_crossing >= (crossing_point - crop_x_min):
            if counting_started:
                rotation_count += 1
                frames_per_rotation.append(frame_count)
                frame_count = 0  # Reset frame count for the next rotation
            else:
                counting_started = True  # Start counting after the first crossing
                frame_count = 0  # Reset frame count for the next rotation
        last_crossing = cx
    
    # Find contours and draw bounding box around the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        cv2.rectangle(cropped_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Add a delay to slow down the video playback
    # time.sleep(0.)  # Adjust the delay as needed (1.0 seconds in this example)

    # Display the number of full rotations completed
    cv2.putText(cropped_frame, f"Rotations: {rotation_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Write the frame to the output video
    out.write(cropped_frame)

    cv2.imshow('Cropped Frame with Annotations', cropped_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# Exclude the last partial rotation
if frames_per_rotation:
    frames_per_rotation.pop()

# Calculate average frames per rotation
if frames_per_rotation:
    average_frames_per_rotation = sum(frames_per_rotation) / len(frames_per_rotation)
    print(f"Average frames per rotation: {average_frames_per_rotation}")
    print(f"Total full rotations: {rotation_count}")
    print(f"Total frames for full rotations: {sum(frames_per_rotation)}")
    
    # Check for outliers
    deviations = [abs(fr - average_frames_per_rotation) for fr in frames_per_rotation]
    max_deviation = max(deviations)
    if max_deviation > average_frames_per_rotation * 0.1:  # 10% deviation threshold
        print(f"Warning: There are outliers in the frames per rotation data. Max Deviation: {max_deviation}")
        print(f"Outliers: {deviations}")
        print(f"frames_per_rotation: {frames_per_rotation}")
    else:
        print(f"All frames per rotation are within acceptable range. Max Deviation: {max_deviation}")
        print(f"frames_per_rotation: {frames_per_rotation}")
else:
    print("No full rotations detected.")



# Frames per second of the camera
fps = 175.4  # Adjust this value based on your camera's FPS

# Calculate rotations per minute (RPM)
if frames_per_rotation:
    total_time_seconds = sum(frames_per_rotation) / fps
    rotations_per_minute = (rotation_count / total_time_seconds) * 60
    print(f"Rotations per minute (RPM): {rotations_per_minute:.2f}")
else:
    print("No full rotations detected, cannot calculate RPM.")


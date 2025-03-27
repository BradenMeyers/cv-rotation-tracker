import cv2
import matplotlib.pyplot as plt
from config import *
from utils import *

# Open the video file
cap = cv2.VideoCapture(video_path)

# Ensure that frames_to_test is not greater than frames_per_rotation
if frames_to_test > frames_per_rotation:
    raise ValueError("frames_to_test cannot be greater than frames_per_rotation")

# Calculate the step size to evenly sample frames across the rotation
step_size = frames_per_rotation // frames_to_test

for i in range(frames_to_test):
    # Calculate the frame index to read
    frame_index = i * step_size

    # Set the video capture to the specific frame index
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    # Read the frame
    ret, frame = cap.read()

    frame = crop_circle(frame)

    # Check if the frame was successfully read
    if not ret:
        print(f"Failed to read frame {i}")
        break

    # Convert the frame from BGR to HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a copy of the frame for drawing contours
    rgb_frame = frame.copy()

    mask = detect_orange_object(frame)

    # Find contours of the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the blank image
    cv2.drawContours(rgb_frame, contours, -1, (0, 255, 0), 1)  # Green color, 1px thick

    # Convert the contour image to RGB for Matplotlib
    contour_image_rgb = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)

    # Plot the RGB and HSV images side by side using Matplotlib
    fig, axs = plt.subplots(1, 3, figsize=(15, 8))

    # Plot the RGB image
    axs[0].imshow(contour_image_rgb)
    axs[0].set_title(f'RGB Image of Frame {i}')
    axs[0].axis('off')  # Hide the axis

    # Plot the HSV image
    axs[1].imshow(hsv_frame)
    axs[1].set_title(f'HSV Image of Frame {i}')
    axs[1].axis('off')  # Hide the axis

    # Plot the mask
    axs[2].imshow(mask, cmap='gray')
    axs[2].set_title(f'Mask of Frame {i}')
    axs[2].axis('off')  # Hide the axis

    plt.show()

# Release the video capture object
cap.release()
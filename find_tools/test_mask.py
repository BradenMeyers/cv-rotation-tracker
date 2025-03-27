import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

from config import *
from utils import *

# Open the video file
cap = cv2.VideoCapture(video_path)

# List to store the pixel counts for each frame
pixel_counts = []

for i in range(frames_per_rotation):
    # Read the frame
    ret, frame = cap.read()
    frame = crop_circle(frame)

    # Check if the frame was successfully read
    if not ret:
        print(f"Failed to read frame {i}")
        break

    mask = detect_orange_object(frame)
    centroid = find_centroid(mask)

    # Count the number of non-zero pixels in the mask
    pixel_count = cv2.countNonZero(mask)
    pixel_counts.append(pixel_count)

    # Create a masked image with the original colors
    masked_image = cv2.bitwise_and(frame, frame, mask=mask)

    # Draw the centroid if detected
    if centroid:
        cx, cy = centroid
        cv2.circle(masked_image, (cx, cy), 5, (0, 255, 0), -1)

    # Show the masked image
    cv2.imshow('Masked Image', masked_image)
    cv2.waitKey(10)
    if delay_video:
        time.sleep(video_delay)

# Release the video capture object
cap.release()
cv2.destroyAllWindows()

# Plot a histogram of the pixel counts
plt.hist(pixel_counts, bins=20, color='blue', edgecolor='black')
plt.title('Histogram of Pixel Counts in Mask')
plt.xlabel('Pixel Count')
plt.ylabel('Frequency')
plt.show()

# Calculate and print the standard deviation of the pixel counts
mean = np.mean(pixel_counts)
print(f"Mean Pixel Count: {mean}")
std_dev = np.std(pixel_counts)
print(f"Standard Deviation of Pixel Counts: {std_dev}")
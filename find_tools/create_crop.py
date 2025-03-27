import cv2
import matplotlib.pyplot as plt
from config import *
from utils import *


# Load the video
cap = cv2.VideoCapture(video_path)

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Read the first frame
ret, frame = cap.read()

# Release the video capture object
cap.release()

if not ret:
    print("Error: Could not read the first frame.")
    exit()

# Convert the frame from BGR to RGB
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Display the frame
plt.imshow(frame_rgb)
plt.title('First Frame')
plt.axis('off')  # Hide the axis
plt.show()


# Ask the user for cropping bounds
x_start = int(input("Enter the starting x coordinate for cropping: "))
x_end = int(input("Enter the ending x coordinate for cropping: "))
y_start = int(input("Enter the starting y coordinate for cropping: "))
y_end = int(input("Enter the ending y coordinate for cropping: "))

# Crop the image
cropped_frame = frame_rgb[y_start:y_end, x_start:x_end]

# Display the video using the cropped frame:
cap = cv2.VideoCapture(video_path)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    cropped_frame = crop(frame)
    cv2.imshow('Cropped Video', cropped_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# # Display the cropped frame
# plt.imshow(cropped_frame)
# plt.title('Cropped Frame')
# plt.axis('off')  # Hide the axis
# plt.show()

# Ask if the cropping looks good
response = input("Does the cropping look good? (y/n): ").strip().lower()

if response == 'y':
    print(f"Cropping bounds - x: ({x_start}, {x_end}), y: ({y_start}, {y_end})")
else:
    print("Please run the script again to adjust the cropping bounds.")
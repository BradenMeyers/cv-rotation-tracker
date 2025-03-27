import cv2
import numpy as np

# Function to detect the orange object
def detect_orange_object(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([10, 100, 100])
    upper_orange = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    return mask

# Function to find the centroid of the detected object
def find_centroid(mask):
    moments = cv2.moments(mask)
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        return cx, cy
    return None

# Initialize video capture
cap = cv2.VideoCapture('Videos/IMG_0860.mov')

# Define the point to check crossing
crossing_point = 900  # Example: vertical line at x=320
cross_count = 0
previous_side = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    mask = detect_orange_object(frame)
    centroid = find_centroid(mask)

    if centroid:
        cx, cy = centroid
        current_side = 'left' if cx < crossing_point else 'right'

        if previous_side and current_side != previous_side:
            cross_count += 1
            previous_side = current_side
        elif not previous_side:
            previous_side = current_side

        # Draw the crossing line and centroid
        cv2.line(frame, (crossing_point, 0), (crossing_point, frame.shape[0]), (255, 0, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f'The object crossed the line {cross_count} times.')
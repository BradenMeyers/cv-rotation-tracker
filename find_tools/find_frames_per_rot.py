import cv2
import time
from config import *

def main():
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot read the frame.")
            break

        # Resize the frame to fit the screen
        frame = cv2.resize(frame, (screen_width, screen_height))

        # Display the frame
        cv2.imshow('Video', frame)
        frame_count += 1

        # Wait for 0.1 seconds (100 ms)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

    print(f"Total frames displayed: {frame_count}")

if __name__ == "__main__":
    main()
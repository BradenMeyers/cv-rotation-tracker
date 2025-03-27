import numpy as np

# TODO shine a light on the object to get a better centroid
lower_orange = np.array([10, 115, 150])
upper_orange = np.array([20, 255, 255])


crop_x_min, crop_x_max = 750, 1200
crop_y_min, crop_y_max = 500, 910

# circular_crop = True
center = (238, 202) 
radius = 211

erode_iter = 1

video_path = 'FinalData/Videos/1.0_58.mov'

fps = 240.12

display_video = True

delay_video = True
video_delay = 0.1

frames_per_rotation = 281  # Approximate number of frames per rotation
frames_to_test = 20  # Number of frames to test for HSV thresholding

# Get screen dimensions
screen_width = 1280  # Replace with your screen width
screen_height = 650  # Replace with your screen height
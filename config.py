import yaml
import numpy as np

## GENERAL SETTINGS
video_delay = 0.1
display_video = False
delay_video = False
frames_to_test = 2  # Number of frames to test for HSV thresholding
degrees = True

# Get screen dimensions
screen_width = 1280  # Replace with your screen width
screen_height = 650  # Replace with your screen height

# Path to the YAML file
yaml_enabled = True
yaml_file_path = 'FinalData/data.yaml'
run_number = 2   # Change this to select a different run

# TODO shine a light on the object to get a better centroid
if yaml_enabled:
    with open(yaml_file_path, 'r') as file:
        data = yaml.safe_load(file)
    
    # Select the desired run number (change as needed)
    config = next((cfg for cfg in data['configs'] if cfg['run_number'] == run_number), None)

    if config:
        lower_orange = np.array(config['lower_orange'])
        upper_orange = np.array(config['upper_orange'])

        crop_x_min, crop_x_max = config['crop_x']
        crop_y_min, crop_y_max = config['crop_y']

        center = tuple(config['center'])
        pixel_center = (int(center[0]), int(center[1]))
        radius = config['radius']

        erode_iter = config['erode_iter']
        video_path = config['video_path']
        fps = config['fps']
        frames_per_rotation = config['frames_per_rotaiton']
    else:
        raise ValueError(f"Run number {run_number} not found in YAML file")

else:
    lower_orange = np.array([10, 115, 150])
    upper_orange = np.array([20, 255, 255])

    crop_x_min, crop_x_max = 750, 1200
    crop_y_min, crop_y_max = 500, 910

    center = (238, 202) 
    radius = 205

    erode_iter = 1
    video_path = 'FinalData/Videos/1.2_55.mov'
    fps = 240.12
    frames_per_rotation = 298  # Approximate number of frames per rotation

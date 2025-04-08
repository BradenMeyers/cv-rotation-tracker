# Calculating Rotation Per Minute Using OPENCV - BYU ME362 

## Project Overview
This was my final project for ME 362 at BYU. OpenCV python code was written to track a orange peice of tape on a vertical axis wind turbine with very high precision (error of les than 0.03 rpm at 450 RPM). This code is a good introduction to the utility of OPENCV and can be applied in many different ways

## Table of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Contributors](#contributors)

<!-- ## Introduction -->


<!-- ## Requirements
- TODO [List any software, libraries, or tools required to run the project.]
- [Example: Python 3.8+, NumPy, Matplotlib] -->

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/BradenMeyers/cv-rotation-tracker.git
    ```
2. Navigate to the project directory:
    ```bash
    cd cv-rotation-tracker
    ```
3. Install dependencies:
    ```bash
    [Provide installation commands, e.g., pip install -r requirements.txt]
    ```

## Usage
Follow allong in the Juptyer notebook for an introduction to the code and then experiment with the tools and parameters to get it to work for your own video!

## Tips
- Always record at the highest frame rate possible for best results. A lower frame rate than 
    - The frame rate (fps) required to record a given RPM is aproxiamtely: fps > (rpm/30)

    - The max RPM that can be recorded given a frame rate is: rpm = fps * 30

- Try to keep a consistent lighting
- Recording from directly above the center will produce higher accuracy instantaneous RPM calculations


## Contributors
- [Braden Meyers](https://github.com/BradenMeyers)
- Project Group Memebers (VAWT Design and Data Collection)- Kevan Williams, Collin Christensen


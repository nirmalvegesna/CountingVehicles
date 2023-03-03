# CountingVehicles

## Overview

Count Vehicles in video using YOLO for training and DeepSORT for tracking.
Vehicles considered are cars, buses, trucks, and motorbikes.

To run the counter and tracker execute the notebook yolo_DS_run_colab.ipynb
This notebook executes yolo_ds.py with the necessary parameters to get an output avi file that displays both real time tracking and the count of vehicles.

## Prerequisite

Download yolo3 weights using wget https://pjreddie.com/media/files/yolov3.weights and add to yolo-coco directory.



## Dependencies

Python3, opencv-python==3.4.2.17, imutils, scipy, tensorflow


## Examples

Some sample input videos are in this directory such as CarsOnHighway.mp4.
The output log files are also provided with the corresponding input video.
The output videos for the provided input videos can be found in this google drive link:https://drive.google.com/drive/folders/1SLZoTJnNeUGFhbp4minF_f327Aq62vHV?usp=share_link


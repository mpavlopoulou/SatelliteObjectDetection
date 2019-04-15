# Object Detection on Satellite Imagery

Object Identification on satellite images using Neural Networks with Tensorflow. Imagery will be
retrieved from ESA’s or NASA’s portals. The model will be fed with training datasets of vehicles,
urban area buildings, and green areas, and will have the ability to identify the boundaries of
these objects. Technologies to use: Python, Keras (Deep Learning python library), and
Tensorflow.

## Prerequisites & Installation

### RetinaNet for object detection (1st Approach)
![Alt text](./images/RetinaNet_Architecture.png?raw=true "RetinaNet Network Architecture")

The one-stage RetinaNet network architecture uses a Feature Pyramid Network (FPN) backbone on top of a feedforward ResNet architecture (a) to generate a rich, multi-scale convolutional feature pyramid (b). To this backbone RetinaNet attaches two subnetworks, one for classifying anchor boxes (c) and one for regressing from anchor boxes to ground-truth object boxes (d). The network design is intentionally simple, which enables this work to focus on a novel focal loss function that eliminates the accuracy gap between our one-stage detector and state-of-the-art two-stage detectors like Faster R-CNN with FPN while running at faster speeds.

1. Install [tensorflow](https://github.com/tensorflow/tensorflow) as per your system requirements.

2. a.Install [keras-retinanet](https://github.com/fizyr/keras-retinanet) a Keras implementation of RetinaNet object detection.
<br>
OR
<br>
b. Install [Mask_RCNN](https://github.com/matterport/Mask_RCNN) an implementation of Mask R-CNN on Python 3, Keras, and TensorFlow.

### YOLT/SIMRDWN for object detection (2nd Approach)
YOLT is an extension of the YOLO v2 framework that can evaluate satellite images of arbitrary size, and runs at ~50 frames per second. Current applications include vechicle detection (cars, airplanes, boats), building detection, and airport detection.

1. Install [tensorflow](https://github.com/tensorflow/tensorflow) as per your system requirements.

2. a.Install [simrdwn](https://github.com/CosmiQ/simrdwn) a rapid satellite imagery object detection framework.
<br>
OR
<br>
b. Install [yolt](https://github.com/CosmiQ/yolt) a framework for Rapid Multi-Scale Object Detection In Satellite Imagery.


**Acknowledgments**<br>
This project was developed during the Spring Semester of 2019 for the module M111 - Big Data.

**Authors**<br>
Maria-Evangelia Pavlopoulou<br>
Georgios Kalampokis



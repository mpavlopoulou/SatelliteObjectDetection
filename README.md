# Object Detection on Satellite Imagery



## Prerequisites & Installation


### RetinaNet for object detection (1st Approach)
![Alt text](./images/RetinaNet_Architecture.png?raw=true "RetinaNet Network Architecture")

The one-stage RetinaNet network architecture uses a Feature Pyramid Network (FPN) backbone on top of a feedforward ResNet architecture (a) to generate a rich, multi-scale convolutional feature pyramid (b). To this backbone RetinaNet attaches two subnetworks, one for classifying anchor boxes (c) and one for regressing from anchor boxes to ground-truth object boxes (d). The network design is intentionally simple, which enables this work to focus on a novel focal loss function that eliminates the accuracy gap between our one-stage detector and state-of-the-art two-stage detectors like Faster R-CNN with FPN while running at faster speeds.

1. Install [tensorflow](https://github.com/tensorflow/tensorflow) as per your system requirements.

2a. Install [keras-retinanet](https://github.com/fizyr/keras-retinanet) a Keras implementation of RetinaNet object detection.

OR

2b. Install [Mask_RCNN](https://github.com/matterport/Mask_RCNN) an implementation of Mask R-CNN on Python 3, Keras, and TensorFlow.

### 


**Acknowledgments**

This project was developed during the Spring Semester of 2019 for the module M111: Big Data.

**Authors**

Maria-Evangelia Pavlopoulou

Georgios Kalampokis



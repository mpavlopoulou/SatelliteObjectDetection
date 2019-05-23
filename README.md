# Object Detection on Satellite Imagery

Object Identification on satellite images using Neural Networks with Tensorflow. 
Datasets used: COWC, SpaceNet Buildings Dataset.
SIMRDWN framework was used for our experiments.
Technologies used: Python, Keras (Deep Learning python library), and
Tensorflow.
Objective: The model is trained upon annotated images containing cars and building footprints,
giving the ability to identify their boundaries.

## Prerequisites & Installation

### YOLT/SIMRDWN for object detection
![Alt text](./images/12TVL160660-CROP_thresh=0.2_road_1024.png?raw=true "RetinaNet Network Architecture")


YOLT is an extension of the YOLO v2 framework that can evaluate satellite images of arbitrary size, and runs at ~50 frames per second. Current applications include vechicle detection (cars, airplanes, boats), building detection, and airport detection.

1. Install [SIMRDWN v1](https://github.com/CosmiQ/simrdwn/tree/v1) as per your system requirements. Follow the installation instructions on the link.
2. Download [COWC](ftp://gdo152.ucllnl.org/cowc/datasets/) dataset. Use ground truth sets as input images.<br>
3. Download [SpaceNet](https://spacenetchallenge.github.io/datasets/datasetHomePage.html) building footprints dataset. 
Needs a AWS account. Suggested client browser: [CloudBerry Explorer for Amazon S3](https://www.cloudberrylab.com/explorer/amazon-s3.aspx)
4. Follow [script_cmds.sh](./script_cmds.sh) commands for reproducing our experiments. In [results](./results) folder you will find all the logs of our trains and tests.

[simrdwn_v1-help.txt](./simrdwn_v1-help.txt) and [simrdwn_v2-help.txt](./simrdwn_v2-help.txt) 
contains a complete list of parameters used by SIMRDWN framework.

**Acknowledgments**<br>
This project was developed during the Spring Semester of 2019 for the module M111 - Big Data.

**Authors**<br>
Maria-Evangelia Pavlopoulou<br>
Georgios Kalampokis



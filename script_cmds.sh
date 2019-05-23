#!/bin/bash
# Script containing commands for SIMRDWN

# Clone repo
cd /home/$USER/workspace
# (v1)
git clone --single-branch --branch v1 https://github.com/CosmiQ/simrdwn.git simrdwn_v1
# (v2)
git clone https://github.com/CosmiQ/simrdwn.git

# create a docker image named "simrdwn"
cd ./simrdwn_v1/docker
# (v1)
nvidia-docker build --no-cache -t simrdwn_v1 .
# (v2)
nvidia-docker build --no-cache -t simrdwn .

# Create and run docker container
# (v1)
nvidia-docker run \
-it --privileged \
--name simrdwn_v1_container0 \
-v /home/pankalos/workspace/simrdwn_v1:/simrdwn \
-v /mnt/srv-ws/m111-imagery/cowc:/data/cowc \
simrdwn_v1
# (v2)
# Mount folders 'simrdwn' and 'cowc'
nvidia-docker run \
-it --privileged \
--name simrdwn_container0 \
-v /home/pankalos/workspace/simrdwn:/simrdwn \
-v /mnt/srv-ws/m111-imagery/cowc:/data/cowc \
simrdwn

# List docker images
docker image ls
# Remove docker image
docker rmi simrdwn_v1

# List docker containers
docker container ls --all
# Start docker
nvidia-docker container start simrdwn_container0
nvidia-docker container start simrdwn_v1_container0
docker container restart simrdwn_container0
# Stop docker
docker container stop simrdwn_container0
# Remove docker
docker container rm simrdwn_container0
# Log in docker bash
nvidia-docker container exec -it simrdwn_container0 bash
nvidia-docker container exec -it simrdwn_v1_container0 bash

# Compile Darknet..
# ..and download weights
cd /simrdwn/yolt/input_weights
curl -O https://pjreddie.com/media/files/yolov2.weights 
cd /simrdwn/yolt2/input_weights
curl -O https://pjreddie.com/media/files/yolov2.weights 
cd /simrdwn/yolt3/input_weights
curl -O https://pjreddie.com/media/files/yolov3.weights 

################
# simrdwn v1
# Phase 1: Pre-process data input
# Setting Paths to "prep_data_cowc.py"
cowc_data_dir = '/data/cowc/datasets/ground_truth_sets/'
simrdwn_data_dir = '/simrdwn/data/'
train_out_dir = '/simrdwn/training_datasets/cowc/'
test_out_dir = '/simrdwn/test_images/cowc/'

# Prepare Training Data (v1, py2.7)
python /simrdwn/core/prep_data_cowc.py


# Phase 2: Train
# SIMRDWN Help
python /simrdwn/core/simrdwn.py --help > /simrdwn/simrdwn_v1-help.txt
python /simrdwn/core/simrdwn.py --help > /simrdwn/simrdwn_v2-help.txt
# Plot loss (YOLT)
python /simrdwn/core/yolt_plot_loss.py --res_dir /simrdwn/results/train_yolt_dense_cowc_2019_05_18_20-45-45/logs/
# Plot loss (TensorFlow)
python /simrdwn/core/tf_plot_loss.py --res_dir /simrdwn/results/train_ssd_inception_v2_cowc_2019_05_20_13-20-12/logs/

# YOLT COWC car search
# general settings
# training settings
# YOLT settings
python /simrdwn/core/simrdwn.py \
	--framework yolt \
	--mode train \
	--gpu 1 \
	--single_gpu_machine 1 \
	--outname dense_cowc \
	--label_map_path /simrdwn/data/class_labels_car.pbtxt \
	--weight_dir /simrdwn/yolt/input_weights \
	--weight_file yolov2.weights \
	\
	--yolt_train_images_list_file cowc_yolt_train_list.txt \
	--max_batches 10000 \
	--batch_size 64 \
	--yolt_input_width 544 \
	--yolt_input_height 544 \
	\
	--yolt_object_labels_str car \
	--yolt_cfg_file yolt.cfg  \
	--subdivisions 16

# SSD COWC car search
# general settings
# training settings
# TF api settings
python /simrdwn/core/simrdwn.py \
	--framework ssd \
	--mode train \
	--gpu 1 \
	--single_gpu_machine 1 \
	--outname inception_v2_cowc \
	--label_map_path /simrdwn/data/class_labels_car.pbtxt \
	\
	--max_batches 10000 \
	--batch_size 16 \
	\
	--tf_cfg_train_file /simrdwn/configs/_altered_v0/ssd_inception_v2_simrdwn.config \
	--train_tf_record /simrdwn/data/cowc_train.tfrecord

# Faster R-CNN car search (ResNet 101) [needs at least 12GB Ram!]
# general settings
# training settings
# TF api settings
python /simrdwn/core/simrdwn.py \
	--framework faster_rcnn \
	--mode train \
	--gpu 1 \
	--single_gpu_machine 1 \
	--outname resnet101_cowc \
	--label_map_path /simrdwn/data/class_labels_car.pbtxt \
	\
	--max_batches 10000 \
	--batch_size 16 \
	\
	--tf_cfg_train_file /simrdwn/configs/_altered_v0/faster_rcnn_resnet101_simrdwn.config \
	--train_tf_record /simrdwn/data/cowc_train.tfrecor


# Phase 3: Test
# YOLT vehicle search
# general settings
# yolt settings
# valid settings
# valid plotting
# train_yolt_dense_cowc_2019_05_18_20-45-45
# train_yolt_dense_cowc_2019_05_14_10-28-49
python /simrdwn/core/simrdwn.py \
	--framework yolt \
	--mode valid \
	--gpu 1 \
	--single_gpu_machine 1 \
	--outname dense_cowc \
	\
	--yolt_object_labels_str car \
	--yolt_cfg_file yolt.cfg \
	\
	--train_model_path train_yolt_dense_cowc_2019_05_18_20-45-45 \
	--weight_file yolt_final.weights \
	--use_tfrecords 0 \
	--valid_testims_dir cowc/Utah_AGRC \
	--slice_sizes_str 544 \
	--edge_buffer_valid 1 \
	--slice_overlap 0.1 \
	--valid_box_rescale_frac 1 \
	--valid_slice_sep __ \
	--min_retain_prob=0.15 \
	\
	--plot_thresh_str 0.2 \
	--show_labels 0 \
	--alpha_scaling 1 \
	--n_valid_output_plots 4 \
	--valid_make_legend_and_title 1 \
	--keep_valid_slices 0

# SSD vehicle search
python /simrdwn/core/simrdwn.py \
	--framework ssd \
	--mode valid \
	--gpu 1 \
	--single_gpu_machine 1 \
	--outname inception_v2_cowc \
	--label_map_path /simrdwn/data/class_labels_car.pbtxt \
	\
	--train_model_path train_ssd_inception_v2_cowc_2019_05_20_13-20-12 \
	--use_tfrecords 0 \
	--valid_testims_dir cowc/Utah_AGRC  \
	--slice_sizes_str 544 \
	--edge_buffer_valid 1 \
	--slice_overlap 0.1 \
	--valid_box_rescale_frac 1 \
	--valid_slice_sep __ \
	--min_retain_prob=0.15 \
	\
	--plot_thresh_str 0.5 \
	--show_labels 0 \
	--alpha_scaling 1 \
	--n_valid_output_plots 4 \
	--valid_make_legend_and_title 0 \
	--keep_valid_slices 0


################
# simrdwn v2

# Prepare Training Data (v2, py3.6)
python /simrdwn/core/parse_cowc.py \
	--truth_dir /data/cowc/datasets/ground_truth_sets/Potsdam_ISPRS \
	--simrdwn_data_dir /simrdwn/train_data/ \
	--out_dir /simrdwn/training_datasets/cowc/ \
	--annotation_suffix _Annotated_Cars.png \
	--category car \
	--input_box_size 10 \
	--verbose 1
#--image_dir /simrdwn/test_images/cowc/ \

# Train
# YOLT vechicle search
python /simrdwn/core/simrdwn.py \
	--framework yolt2 \
	--mode train \
	--gpu 1 \
	--single_gpu_machine 1 \
	--nbands 3 \
	--outname dense_1class_vehicles \
	--label_map_path /simrdwn/train_data/class_labels_car.pbtxt \
	--yolt_train_images_list_file cowc_yolt_train_list.txt \
	--yolt_cfg_file ave_dense.cfg \
	--weight_dir /simrdwn/yolt2/input_weights \
	--weight_file yolov2.weights \
	--max_batches 550 \
	--batch_size 64 \
	--subdivisions 16 

# SSD vehicle search
python /simrdwn/core/simrdwn.py \
	--framework ssd \
	--mode train \
	--gpu 1 \
	--single_gpu_machine 1 \
	--outname inception_v2_1class_vehicles \
	--label_map_path /simrdwn/train_data/class_labels_car.pbtxt \
	--tf_cfg_train_file /simrdwn/configs/_orig/ssd_inception_v2_simrdwn.config \
	--train_tf_record /simrdwn/train_data/cowc_train.tfrecord \
	--max_batches 5001 \
	--batch_size 16 

# test
# YOLT vehicle search
python /simrdwn/core/simrdwn.py \
	--framework yolt2 \
	--mode test \
	--gpu 1 \
	--single_gpu_machine 1 \
	--outname dense_1class_vehicles \
	--label_map_path '/simrdwn/train_data/class_labels_car.pbtxt' \
	--weight_file ave_dense_final.weights \
	--yolt_cfg_file ave_dense.cfg \
	--train_model_path '/simrdwn/results/train_yolt2_dense_1class_vehicles_2019_05_13_17-48-23' \
	--testims_dir '/simrdwn/test_images/cowc/Utah_AGRC' \
	--keep_test_slices 0 \
	--test_slice_sep __ \
	--test_make_legend_and_title 0 \
	--edge_buffer_test 1 \
	--test_box_rescale_frac 1 \
	--plot_thresh_str 0.2 \
	--slice_sizes_str 416 \
	--slice_overlap 0.2 \
	--alpha_scaling 1 \
	--show_labels 1 \
	--test_add_geo_coords 0 \
	--save_json 0


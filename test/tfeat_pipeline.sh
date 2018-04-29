#!/bin/bash

data_dir=$1
use_dsp=${2:-0}


rm "$data_dir/database.db"
rm -r "$data_dir/keypoints"
rm -r "$data_dir/matches"
rm "$data_dir/matches.txt"
rm -r "$data_dir/descriptors"

# detect DoG keypoints using colmap & create database.db
# options for colon images when camera calibration is known
# colmap feature_extractor \
# 	--database_path "$data_dir/database.db" \
# 	--image_path "$data_dir/images" \
# 	--ImageReader.camera_model "PINHOLE" \
# 	--ImageReader.single_camera 1 \
# 	--ImageReader.camera_params 374.1583,375.3094,333.3978,273.8418 \
# 	--SiftExtraction.domain_size_pooling "$use_dsp" \

# another set of option: detect DoG keypoints using colmap & create database.db
colmap feature_extractor \
	--database_path "$data_dir/database.db" \
	--image_path "$data_dir/images" \
	--SiftExtraction.domain_size_pooling "$use_dsp" \

# extract the DoG keypoints into .bin files into keypoints
python colmap_export_keypoints.py \
	--database_path "$data_dir/database.db" \
	--output_path "$data_dir/keypoints"

# main function used to calculate tfeat descriptors
# this also calls a matching function in matlab (GPU enabled)
python main_mat.py "$data_dir" --checkpoint "./testmodel.pth"

# delete old descriptor from .db and write the matching result into match.txt
python refresh_database_write_matching.py\
	 --dataset_path "$data_dir"

# import matches into database.db
colmap matches_importer \
	--database_path "$data_dir/database.db" \
	--match_list_path "$data_dir/matches.txt" \
	--match_type "raw"

rm -r "$data_dir/keypoints"
rm -r "$data_dir/matches"
rm "$data_dir/matches.txt"


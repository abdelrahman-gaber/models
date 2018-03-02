#!/bin/bash

#OAR -p gpu='YES' and host='nefgpu14.inria.fr' and cluster='dellt630gpu'
#OAR -l /gpunum=1,walltime=72 
#OAR --name CaltechTesting-SSD-inception

source activate tensorflow 
module load cuda/8.0 
module load cudnn/6.0-cuda-8.0

dataset_path=/data/stars/user/aabubakr/pd_datasets/datasets/caltech/annot-images/images/test
save_path=/data/stars/user/aabubakr/pd_datasets/outputs/tensorflow-detection/CaltechTest-ssd_inception_v2

# The following finds all the leaf folders in the dataset path and stores them in an array
data_folders=( $(find $dataset_path -type d -mindepth 1 -links 2) )

# Now we iterate through all the image folders
for folder in "${data_folders[@]}"
do
    source_folder=$folder
    save_folder=$save_path/${folder#${dataset_path}}
    # Create the folder if one does not exist already
    mkdir -p $save_folder
    python pd_detection.py -images $source_folder -output $save_folder -model ssd_inception_v2_coco_11_06_2017 -thresh 0.0

   # Give proper permissions so that we do not have to face any delays due to the permissions issue.
    chmod -R 770 $save_folder 
done

cp -a $save_path /home/aabubakr/codes/pedestrian-detection-evaluation/CalEval-3.2.1/data-USA/res/



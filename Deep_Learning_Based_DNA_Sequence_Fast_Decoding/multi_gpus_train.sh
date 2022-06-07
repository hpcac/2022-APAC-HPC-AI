#!/bin/bash

# change the path to your own directory
path="/g/data/ik06/stark/NCI_Leopard"
output_path="/g/data/ik06/stark/NCI_Leopard/output"

## To train a single model with two gpus
horovodrun -np 2 --timeline-filename $output_path/unet_timeline.json python3 $path/deep_tf.py -m cnn

## To train multiple models
# array=( cnn unet se_cnn )
# for i in "${array[@]}"
# do
# 	horovodrun -np 2 --timeline-filename $output_path/"$i"_timeline.json python3 "$path"/deep_tf.py -m $i
# done

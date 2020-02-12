#!/bin/bash
model_path=/data/unilm_small_T8_e5_las/pytorch_model.bin
config_path=/data/unilm_small_T8_e5_las/config.json
labels_file=/data/UniNLP_datasets_v2/all_labels.pl
python uninlp/modeling_decoder.py --config_path $config_path --model_path $model_path --label_file $labels_file 


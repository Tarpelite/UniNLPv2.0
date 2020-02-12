#!/bin/bash
data_dir=/data/UniNLP_datasets_v2
model_path=/data/distill_model_v2/model.bin
config_path=/data/distill_model_v2/bert_config.json
output_dir=/data/multi_gpu_debug
export CUDA_VISIBLE_DEVICES=0
python examples/run_mtdnn.py --model_type bert --model_name_or_path $model_path --output_dir $output_dir --dataset_dir $data_dir --config_name $config_path --tokenizer_name bert-base-uncased --max_seq_length 128 --do_train --do_eval --mini_batch_size 32 --warmup_steps 200 --save_steps 500 --num_train_epochs 1.0 --recover_path $output_dir

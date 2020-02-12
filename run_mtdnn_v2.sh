#!/bin/bash
data_dir=/data/UniNLP_datasets_v2
model_path=bert-base-multilingual-uncased
config_path=bert-base-multilingual-uncased
output_dir=/data/mbert_base_T8_e5_las
export CUDA_VISIBLE_DEVICES=0
python examples/run_mtdnn_v2.py --model_type bert --model_name_or_path $model_path --output_dir $output_dir --dataset_dir $data_dir --config_name $config_path --tokenizer_name $model_path --do_lower_case --max_seq_length 128 --do_train  --do_eval  --mini_batch_size 32 --warmup_steps 88 --save_steps 1000 --num_train_epochs 5.0 

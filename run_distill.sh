#!/bin/bash
dataset_dir=/data/UniNLP_datasets_v2/
model_name_or_path=/data/model_release_v2/unilm_small_T6_e5/onto_ner-ft.bin
config_name=/data/model_release_v2/unilm_small_T6_e5/config.json
output_dir=/data/distill_test_out_pos
teacher_model_name_or_path=/data/model_release_v2/unilm_base_T6_e10/onto_ner-ft.bin
teacher_config_name=/data/model_release_v2/unilm_base_T6_e10/config.json
mini_batch_size=32
max_seq_length=128
gradient_accumulation_steps=1
learning_rate=5e-5
weight_decay=0.01
warmup_steps=500
save_steps=500

python examples/run_distill.py --dataset_dir $dataset_dir --model_type bert --tokenizer_name bert-base-uncased --model_name_or_path $model_name_or_path --config_name $config_name --output_dir $output_dir --do_lower_case --teacher_model_name_or_path $model_name_or_path --teacher_config_name $teacher_config_name --mini_batch_size $mini_batch_size --max_seq_length $max_seq_length --gradient_accumulation_steps $gradient_accumulation_steps --learning_rate $learning_rate --weight_decay $weight_decay --warmup_steps $warmup_steps --save_steps $save_steps

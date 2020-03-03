# docker: hangbo/pytorch:1.2.0-cuda10-apex


export model_path=/mnt/unilm/hangbo/bert-pytorch-dev/model_to_test/model_11_500k.bin
export dataset=/mnt/tianyu/UniNLP/UniNLP_datasets_PARSING_PTB
export output_dir=../unilm_base_single_task_parsing_ptb
export LR=2e-5
export EPOCH=5.0
export WD=0.1
export WR=0.1

export CUDA_VISIBLE_DEVICES=0
python examples/run_mtdnn_v2.py \
  --model_type bert --model_name_or_path $model_path --config_name bert-base-uncased \
  --tokenizer_name bert-base-uncased --do_lower_case --output_dir $output_dir --dataset_dir $dataset \
  --do_eval --do_train  --num_train_epochs $EPOCH --warmup_ratio $WR --save_steps 1000 --max_seq_length 128 \
  --mini_batch_size 32 --learning_rate $LR --weight_decay $WD --gradient_accumulation_steps 1 \
  --fp16 --fp16_opt_level O2

# performance:
# {'PARSING_PTB_UAS': 0.9702072538860104, 'PARSING_PTB_LAS': 0.9421218080450837}

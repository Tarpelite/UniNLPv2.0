export CUDA_VISIBLE_DEVICES=0
data_dir="/data/on_ner_distill/checkpoint-87500"
python examples/run_distill.py --dataset_dir /data/UniNLP_datasets_onto_ner \
--model_type bert --tokenizer_name bert-base-uncased \
--model_name_or_path /data/model_release_v2/unilm_small_T6_e5/onto_ner-ft.bin \
--config_name /data/model_release_v2/unilm_small_T6_e5/config.json \
--output_dir $data_dir  --do_eval \
--do_lower_case  \
--teacher_model_name_or_path /data/model_release_v2/unilm_base_T6_e10/onto_ner-ft.bin \
--teacher_config_name /data/model_release_v2/unilm_base_T6_e10/config.json \
 --mini_batch_size 32 --max_seq_length 128 --num_train_epochs 1000.0\
 --gradient_accumulation_steps 1 --learning_rate 2e-5 \
 --weight_decay 0.01 --warmup_steps 500 --save_steps 500

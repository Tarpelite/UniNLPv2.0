# UniNLPv2.0

This is an Indepent Repo for Clean FrameWork Development

## Installation

```
cd UniNLPv2.0
pip install .
```

## About the UniNLPv2 Datasets Usage

The Directory Tree is like this:
```
UniNLP_datasets_v2/
        config.yaml
        POS/
          train.txt
          dev.txt
          debug.txt
          labels.txt
        NER/
           train.txt
           dev.txt
           debug.txt
           labels.txt
        ....
```

All .txt file(train/dev/debug) use the seq2seq format like
```
   This is an example . \t  Tag1 Tag2 Tag3 Tag4 Tag5
```
Note: the debug.txt is always sample from train.txt, just for debug


The config.yaml contains:

```
tasks:
   - pos
   - ner
   ...
```

The file determines which task will be added and also the order of the tasks


## Run distributed 
```
python -m torch.distributed.launch --nproc_per_node=4 run_mtdnn_ddp.py --dataset_dir /mnt/shaohan/uninlp/UniNLP_datasets_v2_ontoner/ --model_type bert --tokenizer_name bert-base-uncased --model_name_or_path bert-base-uncased --config_name /mnt/shaohan/uninlp/model_release_v2/unilm_base_T6_e10/config.json --output_dir /mnt/shaohan/uninlp/ner_test --do_train  --do_eval --do_lower_case --mini_batch_size 32 --max_seq_length 128 --gradient_accumulation_steps 1 --learning_rate 5e-5 --weight_decay 0.01 --warmup_steps 500 --save_steps 5000
```

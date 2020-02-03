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

## Results

### Single Model For Each Task

|             |POS_UD|POS_ONTO|NER_CONLL|NER_ONTO|Chunking|SRL  |PARSING_PTB|PARSING_UD|AVG  |Size  |    
|-------------|------|--------|---------|--------|--------|-----|-----------|----------|-----|----  |    
|unilm_L6_H384|95.99 |97.08   |88.69    |82.03   |94.77   |82.81|94.16      |84.25     |89.97|87M\*8 |  
|unilm_L2_H384|95.20 |96.58   |83.99    |82.34   |93.14   |67.48|92.10      |78.99     |86.23|63M\*8 |  


### MTDNN Model (joint training only)

|             |POS_UD|POS_ONTO|NER_CONLL|NER_ONTO|Chunking|SRL  |PARSING_PTB|PARSING_UD|AVG  |Size  |  
|-------------|------|--------|---------|--------|--------|-----|-----------|----------|-----|----  |  
|unilm_L12_H768|97.13|97.56   |91.48    |87.70   |96.98   |89.44|94.25      |89.46     |93.00|209M  |  
|unilm_L6_H384|96.10 |97.01   |87.68    |83.85   |96.07   |84.20|94.33      |89.66     |91.11|87M   |   
|unilm_L2_H384|95.67 |96.63   |85.85    |82.31   |95.11   |71.95|92.90      |87.43     |88.48|63M   |  


### MTDNN Model (With Finetuning)

|             |POS_UD|POS_ONTO|NER_CONLL|NER_ONTO|Chunking|SRL  |PARSING_PTB|PARSING_UD|AVG  |Size  |  
|-------------|------|--------|---------|--------|--------|-----|-----------|----------|-----|----  |    
|unilm_L12_H768|97.27|97.54   |91.51    |87.85   |97.13   |89.98|94.40      |90.17     |93.23|209M\*8|  
|unilm_L6_H384|96.42 |97.27   |89.76    |85.61   |96.56   |86.11|94.65      |88.86     |91.91|87M\*8|    
|unilm_L2_H384|95.94 |96.85   |86.61    |83.06   |94.09   |63.38|92.63      |87.20     |87.55|63M\*8|   

### Speed

|Model       |Word Per Second(WPS)|   
| ---        | ------------------ |    
|unilm_2L_cpu| 1986               |  
|unilm_2L_gpu| 3370               |
|unilm_6L_cpu| 1258               |
|unilm_6L_gpu| 1482               |
|spacy_sm    | 1496               |
|spacy_md    | 1352               |
|spacy_lg    | 1431               |  





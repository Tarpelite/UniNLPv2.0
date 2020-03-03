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
## Benchmark

### NLP Progress

|Task                                |Type| Score |  Parper/Source                                              |
|------------------------------------|----|------ |  --------------------------------------------------------   |
|Universe Dependency (Part-of-speech)|Acc |96.77  | [Multilingual BERT and BPEmb (Heinzerling and Strube, 2019)](https://arxiv.org/abs/1906.01569)|  
|Penn Treebank (Part-of-speech)      |Acc |97.96  | [Meta BiLSTM (Bohnet et al., 2018)](https://arxiv.org/abs/1805.08237)              |
|CoNLL 2003(Named Entity Recognition)|F1  |93.50  | [CNN Large + fine-tune (Baevski et al., 2019)](https://arxiv.org/pdf/1903.07785.pdf)|
|OntoNotes v5(Named Entity Recognition)|F1|89.71  | [Flair embeddings(Akbik et al., 2018)](http://aclweb.org/anthology/C18-1139)|
|Penn Treebank (Chunking)|F1|96.72|[Flair embeddings(Akbik et al., 2018)](http://aclweb.org/anthology/C18-1139)|
|OntoNotes v5(Semantic Role Labelling)|F1 |85.50  | [He et al., (2018) + ELMO](http://aclweb.org/anthology/P18-2058)|
|Penn Treebank (Dependency Parsing)  |UAS |97.33  | [Label Attention Layer + HPSG + XLNet (Mrini et al., 2019)](https://khalilmrini.github.io/Label_Attention_Layer.pdf)|
|Universal Dependencies (Dependency Parsing) | UAS |95.80| [Stack-only RNNG (Kuncoro et al., 2017)](https://arxiv.org/abs/1611.05774) |

### Toolkits
|                |POS_UD|POS_ONTO|NER_CONLL|NER_ONTO|Chunking|SRL  |PARSING_PTB|PARSING_UD |    
|----------------|------|--------|---------|--------|--------|-----|-----------|-----------|
|stanford_corenlp|94.94 |-       |89.38    |-       |-       |-    |-          |84.53/79.45| 
|spacy_en_sm     |88.23 |96.76   |-        |84.19   |-       |-    |91.62/89.71|-          |  
|spacy_en_md     |88.53 |96.87   |-        |83.30   |-       |-    |91.93/90.09|-          |
|spacy_en_lg     |88.83 |96.93   |-        |83.38   |-       |-    |92.01/90.17|-          |



## Results

### Single Model For Each Task

|                 |POS_UD|POS_ONTO|NER_CONLL|NER_ONTO|Chunking|SRL  |PARSING_PTB|PARSING_UD |AVG  |Size   |    
|-----------------|------|--------|---------|--------|--------|-----|-----------|---------- |-----|----   |
|unilm_L12_H768   |97.17 |97.49   |92.60    |87.14   |96.62   |88.89|95.21/93.14|93.60/93.14|     |109M\*8|
|unilm_L12_H384   |96.82 |97.36   |91.61    |85.69   |96.17   |87.46|96.63/92.83|97.09/93.60|     |33M\*8 |       
|bert_base_uncased|97.03 |97.25   |90.89    |83.81   |96.27   |87.97|96.11/93.67|95.34/92.25|93.08|109M\*8|
|unilm_L6_H384    |95.99 |97.08   |88.69    |82.03   |94.85   |84.83|94.55/91.12|91.86/88.68|90.29|23M\*8 |  
|unilm_L2_H384    |95.99 |96.87   |86.72    |82.83   |94.28   |73.51|92.10/90.09|93.02/88.88|86.23|16M\*8 |  


### MTDNN Model (joint training + distill)

|                 |POS_UD|POS_ONTO|NER_CONLL|NER_ONTO|Chunking|SRL  |PARSING_PTB|PARSING_UD |AVG  |Size  |  
|-----------------|------|--------|---------|--------|--------|-----|-----------|---------- |-----|----  |  
|unilm_L12_H768   |96.91 |97.28   |90.58    |86.67   |97.06   |88.37|95.33/92.48|93.60/89.93|93.25|109M  |
|unilm_L12_H384   |96.91 |97.17   |91.22    |86.82   |97.25   |88.83|95.08/92.53|91.86/87.65|93.23|33M  |
|unilm_L6_H384    |96.50 |97.13   |89.80    |85.25   |96.70   |86.69|97.15/94.54|93.89/92.29|92.89|23M   |   
|unilm_L2_H384    |96.04 |96.77   |86.10    |82.94   |96.15   |75.02|95.34/92.08|93.60/91.11|88.84|16M   |  
|bert-base-uncased|96.67 |97.17   |90.07    |85.61   |96.92   |86.83|96.24/94.50|95.93/92.32|93.18|109M  |


### MTDNN Model (With Finetuning + distill)

|                 |POS_UD|POS_ONTO|NER_CONLL|NER_ONTO|Chunking|SRL  |PARSING_PTB|PARSING_UD |AVG  |Size   |  
|-----------------|------|--------|---------|--------|--------|-----|-----------|-----------|-----|-------|    
|unilm_L12_H768   |97.34 |97.44   |91.80    |87.61   |97.51   |89.48|95.95/93.98|93.60/88.12|93.27|109M\*8|
|unilm_L6_H384    |96.55 |97.13   |90.00    |85.67   |96.79   |86.48|95.98/93.45|94.48/92.29|92.81|23M\*8 |    
|unilm_L2_H384    |96.35 |96.90   |87.73    |83.11   |95.77   |74.99|95.73/92.81|94.48/91.34|90.50|16M\*8 |
|bert-base-uncased|96.85 |97.27   |90.01    |86.35   |97.13   |88.04|96.50/93.98|96.22/92.60|93.54|109M\*8|
  
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
|CoreNLP_java_vm| 18404           |
|CoreNLP_python_interface|412     |

### Acceleration Rate

#### cpu
|Model | POS | NER | PARSING |
|------|-----|-----|---------|
|base  |  1x | 1x  | 1x      |
|small | 5.5x| 5.2x| 3.7x    |
|tiny  |14.9x|13.5x|6.6x     |

#### gpu
|Model | POS | NER | PARSING |
|------|-----|-----|---------|
|base  |  1x | 1x  | 1x      |
|small | 5.2x| 5.1x| 3.9x    |
|tiny  |5.9x |6.2x | 4.0x    |


## MTDNN Model v1.2  (10w steps)

|             |POS_UD|POS_ONTO|NER_CONLL|NER_ONTO|Chunking|SRL  |PARSING_PTB|PARSING_UD | 
|-------------|------|--------|---------|--------|--------|-----|-----------|-----------|  
|1w           |96.88 |97.15   |87.61    |85.74   |96.21   |85.60|95.20/91.64|93.32/88.31| 
|2w           |97.05 |97.40   |90.75    |86.05   |96.89   |87.01|95.47/92.44|94.48/92.17|   
|3w           |97.17 |97.44   |89.73    |86.53   |97.04   |88.40|95.47/93.03|92.15/88.90|
|3w           |96.96 |97.43   |90.45    |86.79   |97.13   |88.12|95.47/92.51|93.32/89.08|
|4w           |96.91 |97.32   |90.92    |87.01   |97.15   |88.55|95.47/92.44|94.48/88.47|
|5w           |96.96 |97.30   |90.72    |86.73   |97.30   |88.68|95.08/92.00|94.18/89.01|
|6w           |96.92 |97.30   |90.52    |87.00   |97.34   |88.57|95.07/92.71|93.60/88.90|
|7w           |96.76 |97.26   |90.72    |86.73   |97.30   |88.68|95.08/92.01|94.19/89.01|
|8w           |96.96 |97.24   |90.72    |86.73   |97.30   |88.68|95.08/92.00|94.19/89.01|
|9w           |96.91 |97.30   |90.52    |87.00   |97.34   |88.58|95.08/92.71|93.60/88.90|
|10w          |96.91 |97.28   |90.53    |87.03   |97.39   |89.03|95.60/93.03|93.02/88.31|
|11w          |96.81 |97.26   |90.97    |87.01   |97.43   |88.84|94.81/92.64|94.19/87.77|



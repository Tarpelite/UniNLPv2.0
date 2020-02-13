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
|OntoNotes v5(Semantic Role Labelling)|F1 |85.50  | [He et al., (2018) + ELMO](http://aclweb.org/anthology/P18-2058)|
|Penn Treebank (Dependency Parsing)  |UAS |97.33  | [Label Attention Layer + HPSG + XLNet (Mrini et al., 2019)](https://khalilmrini.github.io/Label_Attention_Layer.pdf)|
|Universal Dependencies (Dependency Parsing) | UAS |95.80| [Stack-only RNNG (Kuncoro et al., 2017)](https://arxiv.org/abs/1611.05774) |

### Toolkits
|                |POS_UD|POS_ONTO|NER_CONLL|NER_ONTO|Chunking|SRL  |PARSING_PTB|PARSING_UD|    
|----------------|------|--------|---------|--------|--------|-----|-----------|----------|
|stanford_corenlp|94.94 |-       |89.38    |-       |-       |-    |-          |84.53     | 
|spacy_en_sm     |88.23 |96.76   |-        |84.19   |-       |-    |91.62      |-         |  
|spacy_en_md     |88.53 |96.87   |-        |83.30   |-       |-    |91.93      |-         |
|spacy_en_lg     |88.83 |96.93   |-        |83.38   |-       |-    |92.01      |-         |



## Results

### Single Model For Each Task

|             |POS_UD|POS_ONTO|NER_CONLL|NER_ONTO|Chunking|SRL  |PARSING_PTB|PARSING_UD|AVG  |Size  |    
|-------------|------|--------|---------|--------|--------|-----|-----------|----------|-----|----  |    
|unilm_L6_H384|95.99 |97.08   |88.69    |82.03   |94.77   |82.81|94.16      |84.25     |89.97|87M\*8 |  
|unilm_L2_H384|95.20 |96.58   |83.99    |82.34   |93.14   |67.48|92.10      |78.99     |86.23|63M\*8 |  


### MTDNN Model (joint training + distill)

|              |POS_UD|POS_ONTO|NER_CONLL|NER_ONTO|Chunking|SRL  |PARSING_PTB|PARSING_UD |AVG  |Size  |  
|--------------|------|--------|---------|--------|--------|-----|-----------|---------- |-----|----  |  
|unilm_L12_H768|97.02 |97.37   |90.58    |86.67   |97.06   |88.37|95.33/92.48|93.60/89.93|93.25|209M  |  
|unilm_L6_H384 |96.50 |97.13   |89.80    |85.25   |96.70   |86.69|97.15/94.54|93.89/92.29|92.89|87M   |   
|unilm_L2_H384 |96.04 |96.77   |86.10    |82.94   |96.15   |75.02|95.34/92.08|93.60/91.11|88.84|63M   |  


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
|CoreNLP_java_vm| 18404           |
|CoreNLP_python_interface|412     |






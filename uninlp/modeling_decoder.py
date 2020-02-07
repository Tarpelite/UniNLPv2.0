from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random

import numpy as np
import torch

import torch.nn as nn
from torch.optim import Adam

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

import copy 
import requests
from tqdm import *

import time
from uninlp import WEIGHTS_NAME, BertConfig, MTDNNModel, BertTokenizer



TASK_LIST=["POS","NER", "CHUNKING", "SRL", "ONTO_POS", "ONTO_NER", "PARSING_UD", "PARSING_PTB"]
LABELS_LIST = [
    ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"], 
    ['O', 'B-MISC', 'I-MISC', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC'],
    ['O', 'B-NP', 'B-VP', 'B-PP', 'B-ADVP', 'B-SBAR', 'B-ADJP', 'B-PRT', 'B-CONJP', 'B-INTJ', 'B-LST', 'B-UCP', 'I-NP', 'I-VP', 'I-PP', 'I-ADVP', 'I-SBAR', 'I-ADJP', 'I-PRT', 'I-CONJP', 'I-INTJ', 'I-LST', 'I-UCP'],
    ['O', 'B-V', 'B-A1', 'B-A0', 'I-A0', 'I-A1', 'B-AM-LOC', 'I-AM-LOC', 'B-AM-MNR', 'B-A2', 'I-A2', 'B-A3', 'I-AM-MNR', 'B-AM-TMP', 'I-AM-TMP', 'B-A4', 'I-A4', 'I-A3', 'B-AM-NEG', 'B-AM-MOD', 'B-R-A0', 'B-AM-DIS', 'B-AM-EXT', 'B-AM-ADV', 'I-AM-ADV', 'B-AM-PNC', 'I-AM-PNC', 'I-AM-DIS', 'B-R-A1', 'B-C-A1', 'I-C-A1', 'B-R-AM-TMP', 'I-V', 'B-C-V', 'B-AM-DIR', 'I-AM-DIR', 'B-R-A2', 'B-AM-PRD', 'I-AM-PRD', 'I-R-A2', 'B-R-AM-PNC', 'B-C-AM-MNR', 'I-C-AM-MNR', 'I-R-AM-TMP', 'B-AM-CAU', 'B-R-A3', 'B-R-AM-MNR', 'I-AM-CAU', 'I-AM-EXT', 'B-C-A4', 'I-C-A4', 'I-R-A1', 'B-R-AM-LOC', 'I-R-A0', 'B-C-A0', 'I-C-A0', 'B-C-A2', 'I-C-A2', 'B-R-AM-EXT', 'I-R-AM-EXT', 'B-A5', 'I-R-AM-MNR', 'B-C-AM-LOC', 'I-C-AM-LOC', 'I-R-AM-LOC', 'B-C-A3', 'I-C-A3', 'I-AM-NEG', 'B-R-AM-CAU', 'B-R-A4', 'B-C-AM-ADV', 'I-C-AM-ADV', 'B-R-AM-ADV', 'I-R-AM-ADV', 'I-R-A3', 'B-AM-REC', 'B-AM-TM', 'I-AM-TM', 'B-AM', 'I-AM', 'B-C-A5', 'I-C-A5', 'B-C-AM-TMP', 'I-C-AM-TMP', 'B-AA', 'I-AA', 'B-R-AA', 'I-A5', 'I-AM-MOD', 'B-C-AM-EXT', 'I-AM-REC', 'B-C-AM-NEG', 'I-C-AM-EXT', 'I-C-V', 'B-C-AM-DIS', 'I-C-AM-DIS', 'B-C-AM-CAU', 'I-C-AM-CAU', 'I-R-AM-PNC', 'B-R-AM-DIR', 'I-R-AM-DIR', 'B-C-AM-DIR', 'I-C-AM-DIR', 'B-C-AM-PNC', 'I-C-AM-PNC', 'I-R-A4'],
    ['$', '``', "''", ',', '-LRB-', '-RRB-', '.', ':', 'ADD', 'AFX', 'CC', 'CD', 'DT', 'EX', 'FW', 'GW', 'HYPH', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NFP', 'NIL', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', 'XX', '_SP', ''],
    ['O', 'B-PERSON', 'I-PERSON', 'B-NORP', 'I-NORP', 'B-FAC', 'I-FAC', 'B-ORG', 'I-ORG', 'B-GPE', 'I-GPE', 'B-LOC', 'I-LOC', 'B-PRODUCT', 'I-PRODUCT', 'B-EVENT', 'I-EVENT', 'B-WORK_OF_ART', 'I-WORK_OF_ART', 'B-LAW', 'I-LAW', 'B-LANGUAGE', 'I-LANGUAGE', 'B-DATE', 'I-DATE', 'B-TIME', 'I-TIME', 'B-PERCENT', 'I-PERCENT', 'B-MONEY', 'I-MONEY', 'B-QUANTITY', 'I-QUANTITY', 'B-ORDINAL', 'I-ORDINAL', 'B-CARDINAL', 'I-CARDINAL'],
    [],
    []
]

TASK_MAP = {name:num for num, name in enumerate(TASK_LIST)}

class Token(object):
    def __init__(self, text):
        self.text = text
        self.pos_ = None
        self.ner_ = None
        self.onto_pos_ = None
        self.onto_ner_ = None
        self.chunking_ = None
        self.head_ = None
        self.dep_ = None

    def __repr__(self):
        return self.text
    
    def __str__(self):
        return self.text

class uninlp(object):

    def __init__(self):

        self.special_tokens_count = 2
        self.sep_token = "[SEP]"
        self.sequence_a_segment_id = 0
        self.sequence_b_segment_id = 1
        self.cls_token_segment_id = 1
        self.cls_token = "[CLS]"
        self.pad_token = 0
        self.pad_token_segment_id = 0
        self.mask_padding_with_zero = True
        self.model_type = "bert"
        self.labels_pos = []

    def setup_model(self, model_path, config=None, no_cuda=False):
        self.device  =  torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
        config = BertConfig.from_pretrained(config, 
                                            num_labels=2, 
                                            cache_dir=None, 
                                            output_hidden_states=True)
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased",
                                                do_lower_case=True, 
                                                cache_dir=None)


        model = MTDNNModel.from_pretrained(model_path, 
                                        from_tf=bool(".ckpt" in model_path),
                                        config = config,
                                        labels_list=LABELS_LIST,
                                        task_list = TASK_LIST,
                                        do_task_embedding=False,
                                        do_alpha=False,
                                        do_adapter = False,
                                        num_adapter_layers = 2 
                                          )
        
        self.model_config = config
        self.tokenizer = tokenizer
        self.model = model
        self.model.to(self.device)
    
    def tokenize(self, text):
        # tokenize based on word-piece
        return self.tokenizer.tokenize(text)
    
    def do_predict(self, input_text, task, verb=None, max_seq_length=128):
        
        tokens, orig_tokens = self.tokenizer._tokenize_with_orig(input_text)

        if task == "srl":
            verb_tokens = self.tokenizer.tokenize(input_text)

        if task == "srl":
            special_tokens_count = self.special_tokens_count + 1
        else:
            special_tokens_count = self.special_tokens_count
        
        if task == "srl":
            if len(tokens) > max_seq_length - special_tokens_count - len(verb_tokens):
                tokens = tokens[:(max_seq_length - special_tokens_count - len(verb_tokens))]
        else:
            if len(tokens) > max_seq_length - special_tokens_count:
                tokens = tokens[:(max_seq_length - special_tokens_count)]
        
        valid_length = len(tokens)
        tokens += [self.sep_token]
        segment_ids = [self.sequence_a_segment_id]*len(tokens)
        
        tokens = [self.cls_token] + tokens
        segment_ids = [self.cls_token_segment_id] + segment_ids

        if task == "srl":
            tokens += verb_tokens + [self.sep_token]
            segment_ids += [self.sequence_b_segment_id]*len(verb_tokens) + [self.sequence_b_segment_id]
        
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1 if self.mask_padding_with_zero else 0]*len(input_ids)
        
        padding_length = max_seq_length - len(input_ids)

        input_ids += ([self.pad_token] * padding_length)
        input_mask += ([0 if self.mask_padding_with_zero else 1] * padding_length)
        segment_ids += ([self.pad_token_segment_id]*padding_length)

        task_id = TASK_MAP[task.upper()]
        
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        input_mask = torch.tensor([input_mask], dtype=torch.long).to(self.device)
        segment_ids = torch.tensor([segment_ids], dtype=torch.long).to(self.device)

        with torch.no_grad():
            inputs = {
                "input_ids":input_ids,
                "attention_mask":input_mask,
                "task_id":task_id,
                "token_type_ids":segment_ids
            }
            self.model.eval()
            outputs = self.model(**inputs)
        
        # if task.startswith("parsing"):
        #     logits_arc, logits_label = outputs[:2]
            
        
        logits = outputs[0]
        preds = logits.squeeze().detach().cpu().numpy()
        preds = np.argmax(preds, axis=1)[1:valid_length + 1]
        tokens = tokens[1:valid_length + 1]

        results = []
        r_list = []
        orig_tokens = orig_tokens[:len(tokens)]
        orig_token_list = []

        for tk, pred, orig_token in zip(tokens, preds, orig_tokens):
            if tk.startswith("##") and len(r_list) > 0:
                r_list[-1] = r_list[-1] + tk[2:]
            else:
                r_list.append(tk)
                results.append(pred)
                orig_token_list.append(orig_token)

        if not task.upper().startswith("PARSING"):
            label_list = LABELS_LIST[task_id]
            results = [label_list[x] for x in results]

        result_dict = {
            "task":task,
            "token_list":r_list,
            "orig_token_list": orig_token_list,
            "preds":results
        }

        if task == "srl":
            result_dict["verb"] = verb
        
        return result_dict
    
    def analyze(self, input_text):
           
        pos_tag = self.do_predict(input_text, "pos")
        self.tokens = [Token(text) for text in pos_tag["token_list"]]
        for token, pos in zip(self.tokens, pos_tag["preds"]):
            token.pos_ = pos
        
        ner_tag = self.do_predict(input_text, "ner")
        for token, pred in zip(self.tokens,ner_tag["preds"]):
            token.ner_ = pred.split("-")[-1]
        
        onto_pos_tag = self.do_predict(input_text, "onto_pos")
        for token, pred in zip(self.tokens, onto_pos_tag["preds"]):
            token.onto_pos_ = pred
        
        onto_ner_tag = self.do_predict(input_text, "onto_ner")
        for token, pred in zip(self.tokens, onto_ner_tag["preds"]):
            token.onto_ner_ = pred
    
        chunking_tags = self.do_predict(input_text, "chunking")
        for token, pred in zip(self.tokens, chunking_tags["preds"]):
            token.chunking_ = pred.split("-")[-1]

        heads = self.do_predict(input_text, "parsing_ptb")
        for token, pred in zip(self.tokens, heads["preds"]):
            if pred == 0:
                token.head_ = (0, '[ROOT]')
            else:
                token.head_ = (pred-1, self.tokens[pred-1].text)
        
        return self.tokens
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_path", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()
    nlp = uninlp()
    nlp.setup_model(args.model_path, args.config_path, args.no_cuda)
    test_text = "I have a dog and he likes playing with me."
    s = time.time()
    tokens = nlp.analyze(test_text)
    e = time.time()
    print("Time Used: {} s".format(e - s))
    print("**** test POS tag ****")
    print(tokens)
    print([token.pos_ for token in tokens])
    print("**** test NER tag ****")
    print(tokens)
    print([token.ner_ for token in tokens])
    print("**** test ONTO_POS tag ****")
    print(tokens)
    print([token.onto_pos_ for token in tokens])
    print("**** test ONTO_NER tag ****")
    print(tokens)
    print([token.onto_ner_ for token in tokens])
    print("**** test CHUNKING tag ****")
    print(tokens)
    print([token.chunking_ for token in tokens])
    print("**** test Parsing ****")
    print(tokens)
    print([token.head_ for token in tokens])










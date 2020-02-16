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

from nltk.tokenize.punkt import PunktSentenceTokenizer

import time
from uninlp import WEIGHTS_NAME, BertConfig, MTDNNModelV2, BertTokenizer
# from pudb import set_trace
# set_trace()



TASK_LIST=["POS","NER", "CHUNKING", "SRL", "ONTO_POS", "ONTO_NER", "PARSING_UD", "PARSING_PTB"]

TASK_MAP = {name:num for num, name in enumerate(TASK_LIST)}

class Token(object):
    def __init__(self, text):
        self.text = text
        self.pos_ = None
        self.ner_ = None
        self.onto_pos_ = None
        self.onto_ner_ = None
        self.chunking_ = None
        self.head_ptb_ = None
        self.dep_ptb_ = None
        self.head_ud_ = None
        self.dep_ud_ = None

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

    def setup_model(self, model_path, config=None, label_file=None, no_cuda=False):
        self.device  =  torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
        # load labels
        self.labels_list = torch.load(label_file)
        # with open(label_file, "r", encoding="utf-8") as f:
        #     for line in f:
        #         line = line.strip().split("\t")
        #         self.labels_list.append(line)

        config = BertConfig.from_pretrained(config, 
                                            num_labels=2, 
                                            cache_dir=None, 
                                            output_hidden_states=True)
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased",
                                                do_lower_case=True, 
                                                cache_dir=None)


        model = MTDNNModelV2.from_pretrained(model_path, 
                                        from_tf=bool(".ckpt" in model_path),
                                        config = config,
                                        labels_list=self.labels_list,
                                        task_list = TASK_LIST,
                                        do_task_embedding=False,
                                        do_alpha=False,
                                        do_adapter = False,
                                        num_adapter_layers = 2 
                                          )
        
        self.model_config = config
        self.tokenizer = tokenizer
        self.sent_tokenizer = PunktSentenceTokenizer()
        self.model = model
        self.model.to(self.device)

    def tokenize(self, text):
        # tokenize based on word-piece
        return self.tokenizer.tokenize(text)
    
    def do_predict(self, input_text, task, verb=None, max_seq_length=128):
        
        tokens, orig_tokens = self.tokenizer._tokenize_with_orig(input_text)

        if task == "srl":
            verb_tokens = self.tokenizer.tokenize(verb)

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
        
        if task.startswith("parsing"):
            logits_arc, logits_label = outputs[:2]
            preds_arc = logits_arc.squeeze().detach().cpu().numpy()
            preds_label = logits_label.squeeze().detach().cpu().numpy()
            preds_arc = np.argmax(preds_arc, axis=1)[1:valid_length + 1]
            preds_label = np.argmax(preds_label, axis=1)[1:valid_length+1]
            tokens = tokens[1:valid_length + 1]

            results_head = []
            results = []
            token_list = []
            orig_tokens = orig_tokens[:len(tokens)]
            orig_token_list = []

            for tk, pred_head, pred_label, orig_token in zip(tokens, preds_arc, preds_label, orig_tokens):
                if tk.startswith("##") and len(token_list) > 0:
                    token_list[-1] = token_list[-1] + tk[2:]
                else:
                    token_list.append(tk)
                    results_head.append(pred_head)
                    results.append(pred_label)
                    orig_token_list.append(orig_token)
            
            label_list = self.labels_list[task_id]
            results_head = [x for x in results_head]
            results = [label_list[x] for x in results]
            result_dict = {
                "task":task,
                "token_list":token_list,
                "orig_token_list": orig_token_list,
                "heads":results_head,
                "preds":results
            }

            
            
        else:
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

            
            label_list = self.labels_list[task_id]
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
    
    def batchfy_predict(self, input_text, task, verb=None, max_seq_length=128, batch_size=32):
        sentences = self.sent_tokenizer.tokenize(input_text)
        # print(sentences)
        # max_len = max([len(sent.split()) for sent in sentences])
        # max_seq_length = max_seq_length * max(1, int((max_len+1)/ 32))
        all_input_ids = []
        all_input_mask = []
        all_segment_ids = []
        task_id = TASK_MAP[task.upper()]
        all_orig_tokens = []
        for sentence in sentences:
            input_text = sentence
            tokens, orig_tokens = self.tokenizer._tokenize_with_orig(input_text)
            all_orig_tokens.extend(orig_tokens)
            if task == "srl":
                verb_tokens = self.tokenizer.tokenize(verb)

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

            all_input_ids.append(input_ids)
            all_input_mask.append(input_mask)
            all_segment_ids.append(segment_ids)
            
        dataset = TensorDataset(torch.tensor(all_input_ids, dtype=torch.long), 
                                torch.tensor(all_input_mask, dtype=torch.long),
                                torch.tensor(all_segment_ids, dtype=torch.long))
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
        s = time.time()
        all_token_list = [] 
        all_preds_list = []
        all_heads = []

        for batch in tqdm(dataloader):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)
            input_ids = batch[0]
            input_mask = batch[1]
            segment_ids = batch[2]
            with torch.no_grad():
                inputs = {
                    "input_ids":input_ids,
                    "attention_mask":input_mask,
                    "task_id":task_id,
                    "token_type_ids":segment_ids
                }
                outputs = self.model(**inputs)

            if task.startswith("parsing"):
                logits_arc, logits_label = outputs[:2]
                logits_arc = logits_arc.view(-1, max_seq_length, logits_arc.size(-1))
                logits_label = logits_label.view(-1, max_seq_length, logits_label.size(-1))


                
                for logit_arc, logit_label in zip(logits_arc, logits_label):
                    preds_arc = logit_arc.squeeze().detach().cpu().numpy()
                    preds_label = logit_label.squeeze().detach().cpu().numpy()
                    preds_arc = np.argmax(preds_arc, axis=1)[1:valid_length + 1]
                    preds_label = np.argmax(preds_label, axis=1)[1:valid_length+1]
                    tokens = tokens[1:valid_length + 1]

                    results_head = []
                    results = []
                    token_list = []
                    orig_tokens = orig_tokens[:len(tokens)]
                    orig_token_list = []

                    for tk, pred_head, pred_label, orig_token in zip(tokens, preds_arc, preds_label, orig_tokens):
                        if tk.startswith("##") and len(token_list) > 0:
                            token_list[-1] = token_list[-1] + tk[2:]
                        else:
                            token_list.append(tk)
                            results_head.append(pred_head)
                            results.append(pred_label)
                            orig_token_list.append(orig_token)
                
                    label_list = self.labels_list[task_id]
                    results_head = [x for x in results_head]
                    results = [label_list[x] for x in results]

                    all_token_list.extend(token_list)
                    all_preds_list.extend(results)
                    all_heads.extend(results_head)
            
            else:
                logits = outputs[0]
                logits = logits.view(-1, max_seq_length, logits.size(-1))
                for logit in logits:
                    preds = logit.squeeze().detach().cpu().numpy()
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

                    
                    label_list = self.labels_list[task_id]
                    results = [label_list[x] for x in results]

                    all_token_list.extend(r_list)
                    all_preds_list.extend(results)
            
        if task.startswith("parsing"):
            result_dict = {
                "task":task,
                "token_list":all_token_list,
                "orig_token_list":all_orig_tokens,
                "heads":all_heads,
                "preds":all_preds_list
            }
        else:
            result_dict = {
                "task":task,
                "token_list":all_token_list,
                "orig_token_list":all_orig_tokens,
                "preds":all_preds_list
            }
        e = time.time()
        print("time cost:", e-s)
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
        for token, head, dep in zip(self.tokens, heads["heads"], heads["preds"]):
            if head == 0:
                token.head_ptb_ = (0, '[ROOT]')
                token.dep_ptb_ = dep
            elif head > len(self.tokens):
                token.head_ud_ = (0, '[ROOT]')
                token.dep_ptb_ = dep
            else:
                token.head_ptb_ = (head-1, self.tokens[head-1].text)
                token.dep_ptb_ = dep
        
        heads = self.do_predict(input_text, "parsing_ud")
        for token, head, dep in zip(self.tokens, heads["heads"], heads["preds"]):
            if head == 0:
                token.head_ud_ = (0, '[ROOT]')
                token.dep_ud_ = dep
            elif head > len(self.tokens):
                token.head_ud_ = (0, '[ROOT]')
                token.dep_ud_ = dep
            else:
                token.head_ud_ = (head-1, self.tokens[head-1].text)
                token.dep_ud_ = dep
        
        return self.tokens
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_path", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--label_file", type=str)
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()
    nlp = uninlp()
    nlp.setup_model(args.model_path, args.config_path, args.label_file, args.no_cuda)
    test_text = "I have a dog and he likes playing with me."
    s = time.time()
    tokens = nlp.analyze(test_text)
    e = time.time()
    print("Time Used: {} s".format(e - s))
    print("**** test POS tag ****")
    print(tokens)
    print( ([token.pos_ for token in tokens]))
    print("**** test NER tag ****")
    print(tokens)
    print( ([token.ner_ for token in tokens]))
    print("**** test ONTO_POS tag ****")
    print(tokens)
    print( ([token.onto_pos_ for token in tokens]))
    print("**** test ONTO_NER tag ****")
    print(tokens)
    print( ([token.onto_ner_ for token in tokens]))
    print("**** test CHUNKING tag ****")
    print(tokens)
    print( ([token.chunking_ for token in tokens]))
    print("**** test PTB Parsing ****")
    print(tokens)
    print( ([token.head_ptb_ for token in tokens]))
    print( ([token.dep_ptb_ for token in tokens]))
    print("**** test UD Parsing ****")
    print(tokens)
    print([token.head_ud_ for token in tokens])
    print([token.dep_ud_ for token in tokens])

    test_text = "Microsoft Corporation is an American multinational technology company with headquarters in Redmond Washington. It was founded by Bill Gates and Paul Allen on April 4, 1975."
    print("batchfy_test")
    print(nlp.batchfy_predict(test_text, task="pos"))
    print(nlp.batchfy_predict(test_text, task="onto_pos"))
    print(nlp.batchfy_predict(test_text, task="onto_ner"))
    print(nlp.batchfy_predict(test_text, task="parsing_ptb"))












from __future__ import absolute_import, division, print_function, unicode_literals

import os
import logging
import random

import numpy as np
import torch
from tqdm import tqdm, trange

import torch.nn as nn

from transformers import WEIGHTS_NAME, BertConfig, BertTokenizer
from transformers import MTDNNModelv3 as MTDNNModel


class Decoder(object):

    def __init__(self, 
                 special_tokens_count=2,
                 sep_token = "[SEP]",
                 sequence_a_segment_id = 0,
                 sequence_b_segment_id = 1,
                 cls_token_segment_id = 1,
                 cls_token = "[CLS]",
                 pad_token = 0,
                 pad_token_segment_id = 0,
                 mask_padding_with_zero = True,
                 pos_labels_fn="",
                 ner_labels_fn="",
                 chunking_labels_fn="",
                 srl_labels_fn="",
                 no_cuda=False
                 ):
        
        self.special_tokens_count = special_tokens_count
        self.sep_token = sep_token
        self.sequence_a_segment_id = sequence_a_segment_id
        self.sequence_b_segment_id = sequence_b_segment_id
        self.cls_token_segment_id = cls_token_segment_id
        self.cls_token = cls_token
        self.pad_token = pad_token
        self.pad_token_segment_id = pad_token_segment_id
        self.mask_padding_with_zero = mask_padding_with_zero
        self.task_list = {
            "pos":0,
            "ner":1,
            "chunking":2,
            "srl":3
        }
        self.no_cuda = no_cuda

        with open(pos_labels_fn, "r") as f:
            labels_pos = f.read().splitlines()
            if "X" not in labels_pos:
                labels_pos += ["X"]
            self.labels_pos = labels_pos 
        
        with open(ner_labels_fn, "r") as f:
            labels_ner = f.read().splitlines()
            if "O" not in labels_ner:
                labels_ner = ["O"] + labels_ner
            self.labels_ner = labels_ner

        with open(chunking_labels_fn, "r") as f:
            labels_chunking = f.read().splitlines()
            if "O" not in labels_chunking:
                labels_chunking = ["O"] + labels_chunking
            self.labels_chunking = labels_chunking

        with open(srl_labels_fn, "r") as f:
            labels_srl = f.read().splitlines()
            labels_srl = [x for x in labels_srl if len(x) > 0]
            if "O" not in labels_srl:
                labels_srl = ["O"] + labels_srl

            self.labels_srl = labels_srl
    
    def setup_model(self, 
                    config=None,
                    model_name_or_path=None,
                    tokenizer_name=None,
                    do_lower_case=False,
                    ):
        
        print("********** Setting Up Model **********")
        if self.no_cuda:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() and not self.no_cuda else "cpu")
        
        self.config = BertConfig.from_pretrained(config,
                                                num_labels=2,
                                                cache_dir=None,
                                                output_dir=None,
                                                output_hidden_states=True)
        
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name,
                                                       do_lower_case=do_lower_case,
                                                       cache_dir=None)
        
        self.model = MTDNNModel.from_pretrained(model_name_or_path,
                                                from_tf=bool(".ckpt" in model_name_or_path),
                                                config=self.config,
                                                num_labels_pos = len(self.labels_pos),
                                                num_labels_ner = len(self.labels_ner),
                                                num_labels_chunking = len(self.labels_chunking),
                                                num_labels_srl = len(self.labels_srl),
                                                cache_dir = None,
                                                init_last = False)
        
        self.model.to(self.device)
    
    def do_predict(self, input_text, task, verb=None, max_seq_length=128, batch_size=1):
        task = task.lower()
        label_list = getattr(self, "labels_{}".format(task))

        if batch_size == 1:
            input_text = [input_text]

        all_input_ids = []
        all_input_mask = []
        all_segment_ids = []
        all_valid_lengths = []

        for text in input_text:
            # print("text check")
            # print(text)
            tokens = self.tokenizer.tokenize(text)

            if task == "srl":
                verb_tokens = self.tokenizer.tokenize(text)
            
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
                tokens += [self.sep_token] + verb_tokens + [self.sep_token]
                segment_ids += [self.sequence_a_segment_id] + [self.sequence_b_segment_id]*len(verb_tokens) + [self.sequence_b_segment_id]
            
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1 if self.mask_padding_with_zero else 0]*len(input_ids)
            
            padding_length = max_seq_length - len(input_ids)

            input_ids += ([self.pad_token] * padding_length)
            input_mask += ([0 if self.mask_padding_with_zero else 1] * padding_length)
            segment_ids += ([self.pad_token_segment_id]*padding_length)

            task_id = self.task_list[task]

            all_input_ids.append(input_ids)
            all_input_mask.append(input_mask)
            all_segment_ids.append(segment_ids)
            all_valid_lengths.append(valid_length)

            
        if self.no_cuda:
            all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
            all_input_mask = torch.tensor(all_input_mask, dtype=torch.long)
            all_segment_ids = torch.tensor(all_segment_ids, dtype=torch.long)
        else:
            all_input_ids = torch.tensor(all_input_ids, dtype=torch.long).cuda()
            all_input_mask = torch.tensor(all_input_mask, dtype=torch.long).cuda()
            all_segment_ids = torch.tensor(all_segment_ids, dtype=torch.long).cuda()
        
        with torch.no_grad():
            inputs = {
                "input_ids":all_input_ids,
                "attention_mask":all_input_mask,
                "task_id": task_id,
                "token_type_ids":all_segment_ids
            }
            self.model.eval()
            outputs = self.model(**inputs)

        all_results = []
        all_r_list = []
        all_verbs = []
        for outs, valid_length in zip(outputs[0], all_valid_lengths):    
            logits = outs
            # print(logits.shape)
            preds = logits.squeeze().detach().cpu().numpy()
            preds = np.argmax(preds, axis=1)[1:valid_length + 1]
            tokens = tokens[1:valid_length + 1]

            results = []
            r_list = []
            for tk, pred in zip(tokens, preds):
                if tk.startswith("##") and len(r_list) > 0:
                    r_list[-1] = r_list[-1] + tk[2:]
                else:
                    r_list.append(tk)
                    results.append(pred)
            results = [label_list[x] for x in results]
            all_results.append(results)
            all_r_list.append(r_list)


        result_dict = {
            "task":task,
            "token_list":all_r_list,
            "pred_label":all_results
        }
        if task == "srl":
            result_dict['verb'] = verb
        
        # print(result_dict)
        return result_dict

    
if __name__ == "__main__":

    pos_labels_fn = "/mnt/nlpdemo/MSRA_nlp_tool/labels/pos.txt"
    ner_labels_fn = "/mnt/nlpdemo/MSRA_nlp_tool/labels/ner.txt"
    chunking_labels_fn = "/mnt/nlpdemo/MSRA_nlp_tool/labels/chunking.txt"
    srl_labels_fn = "/mnt/nlpdemo/MSRA_nlp_tool/labels/srl.txt"

    no_cuda = False
    config = "/mnt/nlpdemo/MSRA_nlp_tool/model4server/config.json"
    tokenizer_name="/mnt/nlpdemo/MSRA_nlp_tool/model4server/vocab.txt"
    model_path = "/mnt/nlpdemo/MSRA_nlp_tool/model4server/pytorch_model.bin"

    decoder = Decoder(pos_labels_fn = pos_labels_fn,
                      ner_labels_fn = ner_labels_fn,
                      chunking_labels_fn = chunking_labels_fn,
                      srl_labels_fn = srl_labels_fn
                      )
    
    decoder.setup_model(
        no_cuda=False,
        config = config,
        model_name_or_path = model_path,
        tokenizer_name = tokenizer_name,
        do_lower_case=True
    )

    input_text = "I have a cute dog and he likes playing with me."

    pos_results = decoder.do_predict(input_text, "pos")
    ner_results = decoder.do_predict(input_text, "ner")
    chunking_results = decoder.do_predict(input_text, "chunking")
    srl_results = decoder.do_predict(input_text, "srl", verb="have")
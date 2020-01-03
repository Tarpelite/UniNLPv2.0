from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random

import numpy as np
import torch
from seqeval.metrics import precision_score, recall_score, f1_score, accuracy_score
from tensorboardX import SummaryWriter
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange


import torch.nn as nn
from torch.optim import Adam
import copy

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import WEIGHTS_NAME, BertConfig, BertForTokenClassification, BertTokenizer
from transformers import MTDNNModelv3 as MTDNNModel
from transformers import MTDNNModelTaskEmbeddingV2 as TaskEmbeddingModel
from transformers import AdapterMTDNNModel 
from transformers import RobertaConfig, RobertaForTokenClassification, RobertaTokenizer
from transformers import DistilBertConfig, DistilBertForTokenClassification, DistilBertTokenizer

# task_id list: {POS:0, NER:1, Chunking:2, SRL:3}
ALL_MODELS = sum(
    (tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, RobertaConfig, DistilBertConfig)),
    ())

MODEL_CLASSES = {
    "bert":(BertConfig, MTDNNModel, BertTokenizer),
    "task_embedding":(BertConfig, TaskEmbeddingModel, BertTokenizer)
}


class Decoder(object):

    def __init__(self, 
                special_tokens_count = 2,
                sep_token = "[SEP]",
                sequence_a_segment_id = 0,
                sequence_b_segment_id = 1,
                cls_token_segment_id = 1,
                cls_token="[CLS]",
                pad_token=0,
                pad_token_segment_id=0,
                mask_padding_with_zero=True,
                model_type="bert",
                ):
        
        self.MODEL_CLASSES = {
            "bert":(BertConfig, MTDNNModel, BertTokenizer),
            "task_embedding":(BertConfig, TaskEmbeddingModel, BertTokenizer)
        }
        self.special_tokens_count = special_tokens_count
        self.sep_token = sep_token
        self.sequence_a_segment_id = sequence_a_segment_id
        self.sequence_b_segment_id = sequence_b_segment_id
        self.cls_token_segment_id = cls_token_segment_id
        self.cls_token = cls_token
        self.pad_token = pad_token
        self.pad_token_segment_id = pad_token_segment_id
        self.mask_padding_with_zero = mask_padding_with_zero
        self.model_type = model_type
        self.task_list = {
            "pos":0,
            "ner":1,
            "chunking":2,
            "srl":3
        }

    def get_labels(self, pos_labels_fn, ner_labels_fn, chunking_labels_fn, srl_labels_fn):

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
                    no_cuda=False, 
                    config=None, 
                    model_name_or_path=None, 
                    tokenizer_name=None, 
                    do_lower_case=False,
                    labels_pos_fn="",
                    labels_ner_fn="",
                    labels_chunking_fn="",
                    labels_srl_fn=""):
        self.device  =  torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
        
        self.get_labels(labels_pos_fn, labels_ner_fn, labels_chunking_fn, labels_srl_fn)
        config_class, model_class, tokenizer_class = self.MODEL_CLASSES[self.model_type]

        config = config_class.from_pretrained(config,
                                              num_labels=2,
                                              cache_dir=None,
                                              output_hidden_states=True)
        tokenizer = tokenizer_class.from_pretrained(tokenizer_name,
                                                    do_lower_case=do_lower_case,
                                                    cache_dir=None)
        

        model = model_class.from_pretrained(model_name_or_path,
                                            from_tf=bool(".ckpt" in model_name_or_path),
                                            config=config,
                                            num_labels_pos = len(self.labels_pos),
                                            num_labels_ner = len(self.labels_ner),
                                            num_labels_chunking = len(self.labels_chunking),
                                            num_labels_srl= len(self.labels_srl),
                                            cache_dir=None,
                                            init_last=False)
        
        self.model_config = config
        self.tokenizer = tokenizer
        self.model = model

        self.model.to(self.device)
    
    def do_predict(self, input_text, task, verb=None, max_seq_length=128):
        task = task.lower()
        label_list = getattr(self, "labels_{}".format(task))


        tokens = self.tokenizer.tokenize(input_text)

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
            tokens += [self.sep_token] + verb_tokens + [self.sep_token]
            segment_ids += [self.sequence_a_segment_id] + [self.sequence_b_segment_id]*len(verb_tokens) + [self.sequence_b_segment_id]
        
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1 if self.mask_padding_with_zero else 0]*len(input_ids)
        
        padding_length = max_seq_length - len(input_ids)

        input_ids += ([self.pad_token] * padding_length)
        input_mask += ([0 if self.mask_padding_with_zero else 1] * padding_length)
        segment_ids += ([self.pad_token_segment_id]*padding_length)

        task_id = self.task_list[task]
        
        input_ids = torch.tensor([input_ids], dtype=torch.long).cuda()
        input_mask = torch.tensor([input_mask], dtype=torch.long).cuda()
        segment_ids = torch.tensor([segment_ids], dtype=torch.long).cuda()
        
        with torch.no_grad():
            inputs = {
                "input_ids":input_ids,
                "attention_mask":input_mask,
                "task_id": task_id,
                "token_type_ids":segment_ids
            }
            self.model.eval()
            outputs = self.model(**inputs)
            
        logits = outputs[0]
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
        result_dict = {
            "task":task,
            "token_list":r_list,
            "pred_label":results
        }
        if task == "srl":
            result_dict['verb'] = verb
        
        print(result_dict)
        return result_dict


    
        
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default=None, type=str, required=True, 
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    
    parser.add_argument("--labels_pos", default=None, type=str)
    parser.add_argument("--labels_ner", default=None, type=str)
    parser.add_argument("--labels_chunking", default=None, type=str)
    parser.add_argument("--labels_srl", default=None, type=str)

    parser.add_argument("--config_name", default="", type=str)
    parser.add_argument("--tokenizer_name", default="", type=str)
    parser.add_argument("--cache_dir", default="", type=str)
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--no_cuda", action="store_true")
    
    args = parser.parse_args()

    
    input_text = "I have a dog and he likes playing with me."
    
    decoder = Decoder(
                model_type=args.model_type
    )
    decoder.setup_model(
        no_cuda = args.no_cuda,
        config = args.config_name,
        model_name_or_path = args.model_name_or_path,
        tokenizer_name = args.tokenizer_name,
        do_lower_case = args.do_lower_case,
        labels_pos_fn = args.labels_pos,
        labels_ner_fn = args.labels_ner,
        labels_chunking_fn= args.labels_chunking,
        labels_srl_fn = args.labels_srl
    )
    
    results = decoder.do_predict(input_text, task="pos")
    print(results)

if __name__ == "__main__":
    main()

    
    


    


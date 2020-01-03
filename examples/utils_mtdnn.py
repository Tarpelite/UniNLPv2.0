from __future__ import absolute_import, division, print_function

import logging
import os
from io import open
import yaml
from tqdm import *
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

logger = logging.getLogger(__name__)

class MegaDataSet(object):
    def __init__(self, 
                 datasets_dir, 
                 max_seq_length,
                 tokenizer,
                 mini_batch_size):
        with open(os.path.join(datasets_dir, "config.yaml"), "r", encoding="utf-8") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)

        task_list = data["tasks"]
        task_list = [x.upper() for x in task_list]
        labels_list = []

        for task in task_list:
            labels_file = os.path.join(os.path.join(datasets_dir, task, "labels.txt"))
            with open(labels_file, "r", encoding="utf-8") as f:
                labels = f.read().splitlines() # no blank line in the end
                labels_list.append(labels)
        
        self.task_list = task_list
        self.labels_list = labels_list
        self.task_map = {task:idx for idx, task in enumerate(task_list)}
        self.datasets_dir = datasets_dir
        self.features_map =  {}
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.mini_batch_size = mini_batch_size
    
    def load_examples_from_file(self, file_path, task_name):
        task = task_name.upper()
        examples = []
        with open(file_path, encoding="utf-8") as f:
            for line in f.readlines():
                words = []
                labels = []
                inputs = line.strip().strip("\n").split("\t")
                left = inputs[0].strip().split()
                right = inputs[1].strip().split()
                if task == "SRL":
                    words = left
                    labels = right
                    assert len(words) == len(labels) + 1
                    examples.append([words, labels])
                    
                else:
                    words = left
                    labels = right
                    assert len(words) == len(labels)
                    examples.append([words, labels])

        return examples

    def load_single_dataset(self, task_name, mode):
        task = task_name.upper()
        task_id = self.task_map[task_name]
        file_path = os.path.join(self.datasets_dir, task_name, "{}.txt".format(mode))
        print("Loading trainset from task %s".format(task))
        
        examples = self.load_examples_from_file(file_path, task_name)
        labels = self.labels_list[task_id]

        label_map = {label:i for i, label in enumerate(labels)}

        features = []
        cnt_counts = []

        if "{}-{}".format(task, "train") in self.features_map:
            features = self.features_map["{}-{}".format(task, mode)]
        else:
            for (ex_index, example) in enumerate(tqdm(examples)):
                if ex_index % 10000 == 0:
                    logger.info("Writing example %d of %d", ex_index, len(examples))
                tokens = []
                label_ids = []
                
                words = example[0]
                labels = example[1]
                if task_name == "SRL":
                    verb = words[0]
                    words = words[1:]


                assert len(words) == len(labels)
                for word, label in zip(words, labels):
                    word_tokens = self.tokenizer.tokenize(word)
                    tokens.extend(word_tokens)
                    label_ids.extend([label_map[label]] + [-100]*(len(word_tokens) - 1))

                cnt_counts.append(len(tokens))
                if task_name == "SRL":
                    verb_tokens = self.tokenzer.tokenize(verb)
                    special_tokens_count = 3
                    if len(tokens) > self.max_seq_length - special_tokens_count - len(verb_tokens):
                        tokens = tokens[:(self.max_seq_length - special_tokens_count - len(verb_tokens))]
                else:
                    special_tokens_count = 2
                    if len(tokens) > self.max_seq_length - special_tokens_count:
                        tokens = tokens[:(self.max_seq_length - special_tokens_count)]
                
                tokens += ['[SEP]']
                label_ids += [-100]
                segment_ids = [0]*len(tokens)

                tokens = ['[CLS]'] + tokens
                label_ids = [-100] + label_ids
                segment_ids = [0] + segment_ids

                if task_name == "SRL":
                    tokens += ['[SEP]'] + verb_tokens + ['[SEP]']
                    label_ids += [-100] + [-100]*(len(verb_tokens) + 1) 
                    segment_ids += [0]  + [1]*(len(verb_tokens) + 1)
                
                input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1]*len(input_ids)

                padding_length = self.max_seq_length - len(input_ids)

                input_ids += [0]*padding_length
                input_mask += [0]*padding_length
                segment_ids += [0]*padding_length
                label_ids += [0]*padding_length

                assert len(input_ids) == self.max_seq_length
                assert len(input_mask) == self.max_seq_length
                assert len(segment_ids) == self.max_seq_length
                assert len(label_ids) == self.max_seq_length

                if ex_index < 5:
                    logger.info("*** {} Example ***".format(task))
                    logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
                    logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
                    logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
                    logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
                    logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

                features.append([
                    input_ids, input_mask, segment_ids, label_ids
                ])
            

            self.features_map["{}-{}".format(task, "train")] = features

            logger.info("*** Statistics ***")
            logger.info("*** max_len:{}  min_len:{} avg_len:{}***".format(max(cnt_counts), min(cnt_counts), sum(cnt_counts) / len(cnt_counts)))
            all_input_ids = torch.tensor([x[0] for x in features], dtype=torch.long)
            all_input_mask = torch.tensor([x[1] for x in features], dtype=torch.long)
            all_segment_ids = torch.tensor([x[2] for x in features], dtype=torch.long)
            all_label_ids = torch.tensor([x[3] for x in features], dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        return features, dataset
    
    def load_joint_train_dataset(self, debug=False):
        features_batches_list = []
        for task in self.task_list:
            if debug:
                features, _ = self.load_single_dataset(task, "debug")
            else:
                features, _ = self.load_single_dataset(task, "train")
            cnt = 0
            features_batches = []
            while cnt + self.mini_batch_size < len(features):
                batch_t = []
                for i in range(cnt, cnt + self.mini_batch_size):
                    batch_t.append(features[i])
                features_batches.append(batch_t)
                cnt += self.mini_batch_size
            batch_t = []
            for i in range(cnt, len(features)):
                batch_t.append(features[i])
            features_batches.append(batch_t)
            task_id = self.task_map[task]
            features_batches_list.append([features_batches])
        return features_batches_list


    
        

        





            

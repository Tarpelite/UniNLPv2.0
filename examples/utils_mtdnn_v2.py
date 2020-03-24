from __future__ import absolute_import, division, print_function

import logging
import os
from io import open
import yaml
from tqdm import *
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, ConcatDataset, BatchSampler, Sampler
from multiprocessing import cpu_count, Pool
import math
from random import shuffle
import torch.distributed as dist

logger = logging.getLogger(__name__)

def tensor2cuda(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor

def project(x, original_x, epsilon, _type="linf"):
    if _type == "linf":
        max_x = original_x + epsilon
        min_x = original_x - epsilon

        x = torch.max(torch.min(x, max_x), min_x)
    elif _type == 'l2':
        dist = (x - original_x)
        dist = dist.view(x.shape[0], -1)
        dist_norm = torch.norm(dist, dim=1, keepdim=True)
        mask = (dist_norm > epsilon).unsqueeze(2).expand_as(x)
        dist = dist / dist_norm
        dist *= epsilon
        dist = dist.view(x.shape)
        x = (original_x + dist)*mask.float() + x*(1-mask.float())
    else:
        raise NotImplementedError
    return x

class AdversarialAttack:
    def __init__(self, model, epsilon, alpha, min_val, max_val, max_iters, _type='linf'):
        self.model = model

        self.epsilon = epsilon
        self.alpha = alpha
        self.min_val = min_val
        self.max_val = max_val
        self.max_iters = max_iters
        self._type = _type
    
    def perturb(self, inputs, reduction4loss='mean', random_stat=False):
        original_text = inputs["sequence_output"]
        if random_stat:
            rand_perturb = torch.FloatTensor(original_text.shape).uniform_(
                -self.epsilon, self.epsilon
            )
            rand_perturb = tensor2cuda(rand_perturb)
            x = original_text + rand_perturb
            x.clamp_(self.min_val, self.max_val)
        else:
            x = original_text.clone()
        
        x.requires_grad = True
        with torch.enable_grad():
            for _iter in range(self.max_iters):
                self.model.eval()
                inputs["sequence_output"] = x
                outputs = self.model(**inputs)
                loss = outputs[0]
                if reduction4loss == 'none':
                    grad_outputs = tensor2cuda(torch.ones(loss.shape))
                else:
                    grad_outputs = None
                
                grads = torch.autograd.grad(loss, x, grad_outputs=grad_outputs, only_inputs=True)

                grads = grads[0]
                x.data += self.alpha *torch.sign(grads.data)

                x = project(x, original_text, self.epsilon, self._type)
                x.clamp_(self.min_val, self.max_val)
        bias = x - original_text
        return bias



    
def las_score(true_label_list, true_heads_list, predict_label_list, predict_heads_list):
    total_score = 0
    cnt = len(true_label_list)
    for true_labels, true_heads, predict_labels, predict_heads in zip(true_label_list, true_heads_list, predict_label_list, predict_heads_list):
        hit = 0
        for true_label, true_head, predict_label, predict_head in zip(true_labels, true_heads, predict_labels, predict_heads):
            if true_label == predict_label and true_head == predict_head:
                hit += 1
        score = hit*1.000 / len(true_labels)
        total_score += score
    return total_score*1.000/cnt

class RandomBatchSampler(Sampler):
    def __init__(self, data_source, batch_size):

        self.data_source = data_source
        self.batch_size = batch_size
        assert len(self.data_source) % self.batch_size == 0

        self.batch_sampler = list(BatchSampler(SequentialSampler(range(len(self.data_source))),
                                               batch_size=self.batch_size, drop_last=True))

        self.random_id_sampler = torch.randperm(len(self.batch_sampler)).tolist()

    def __iter__(self):
        # when iter, do random
        for ran_id in self.random_id_sampler:
            yield self.batch_sampler[ran_id]

    def __len__(self):
        return len(self.random_id_sampler)


class DistributedRandomBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, num_replicas=None, rank=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.data_source = data_source
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        assert len(self.data_source) % self.batch_size == 0

        self.batch_sampler = list(BatchSampler(SequentialSampler(range(len(self.data_source))),
                                               batch_size=self.batch_size, drop_last=True))

        self.num_samples = int(math.floor(len(self.batch_sampler) * 1.0 / self.num_replicas))

        self.total_size = self.num_samples * self.num_replicas

        # print("sample log --------------", self.total_size, len(self.batch_sampler))
        self.random_id_sampler = torch.randperm(self.total_size).tolist()

        self.rank = rank
        self.epoch = 0

    def __iter__(self):
        # when iter, do random
        for ran_id in self.random_id_sampler[self.rank:self.total_size:self.num_replicas]:
            yield self.batch_sampler[ran_id]

    def __len__(self):
        return len(self.random_id_sampler)


class MegaDataSet(object):
    def __init__(self, 
                 datasets_dir, 
                 max_seq_length,
                 tokenizer,
                 mini_batch_size,
                 sep_token_extra=False):
        with open(os.path.join(datasets_dir, "config.yaml"), "r", encoding="utf-8") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)

        task_list = data["tasks"]
        task_list = [x.upper() for x in task_list]
        labels_list = [] # for Parsing, no labels

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
        self.sep_token_extra = sep_token_extra
    
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
                elif task.startswith("PARSING"):
                    words = inputs[0].strip().split()
                    heads = inputs[1].strip().split()
                    labels = inputs[2].strip().split()
                    examples.append([words, heads, labels])  
                else:
                    words = left
                    labels = right
                    assert len(words) == len(labels)
                    examples.append([words, labels])

        return examples

    def load_single_dataset(self, task_name, batch_size, mode):
        task = task_name.upper()
        task_id = self.task_map[task_name]
        file_path = os.path.join(self.datasets_dir, task_name, "{}.txt".format(mode))
        print("Loading trainset from task {}".format(task))
        
        examples = self.load_examples_from_file(file_path, task_name)
        labels = self.labels_list[task_id]

        label_map = {label:i for i, label in enumerate(labels)}

        features = []
        cnt_counts = []
    
        if task in ["POS" , "NER" , "ONTO_POS" , "ONTO_NER"]:
            last_tokens = []
            last_label_ids = []
            half_length = int(self.max_seq_length / 2)

        if "{}-{}".format(task, mode) in self.features_map:
            features = self.features_map["{}-{}".format(task, mode)]
        else:
            for (ex_index, example) in enumerate(tqdm(examples)):
                # if ex_index % 10000 == 0:
                #     logger.info("Writing example %d of %d", ex_index, len(examples))
                tokens = []
                label_ids = []
                tok_to_orig_index = []
                orig_to_tok_index = []
                head_ids = []
                skip_num = 0 # skip long sentences
                
                words = example[0]
                labels = example[1]
                if task == "SRL":
                    verb = words[0]
                    words = words[1:]
                elif task.startswith("PARSING"):
                    words = example[0]
                    heads = example[1]
                    labels = example[2]
                    assert len(words) == len(labels) == len(heads)

                assert len(words) == len(labels)
                if task.startswith("PARSING"):
                    for word, head, label in zip(words, heads, labels):
                        orig_to_tok_index.append(len(tokens))
                        word_tokens = self.tokenizer.tokenize(word)
                        tokens.extend(word_tokens)
                        if head == "_" or int(head) > (self.max_seq_length - 2):
                            head = -100 #pad_token_label_id
                        
                        head_ids.extend([int(head)] + [-100] * (len(word_tokens) - 1))
                        label_ids.extend([label_map[label]] + [-100]*(len(word_tokens) - 1))
    

                else:    
                    for word, label in zip(words, labels):
                        orig_to_tok_index.append(len(tokens))
                        word_tokens = self.tokenizer.tokenize(word)
                        tokens.extend(word_tokens)
                        label_ids.extend([label_map[label]] + [-100]*(len(word_tokens) - 1))

                if task in ["POS","NER", "ONTO_NER" , "ONTO_POS"]:
                    if ex_index == 0 :
                        last_tokens = tokens[-half_length:]
                        last_label_ids = label_ids[-half_length:]
                    else:
                        tokens = last_tokens + tokens
                        label_ids = last_label_ids + label_ids
                        last_tokens = tokens[-half_length:]
                        last_label_ids = label_ids[-half_length:]

                cnt_counts.append(len(tokens))
                if task == "SRL":
                    verb_tokens = self.tokenizer.tokenize(verb)
                    special_tokens_count = 3
                    if self.sep_token_extra:
                    if len(tokens) > self.max_seq_length - special_tokens_count - len(verb_tokens):
                        tokens = tokens[:(self.max_seq_length - special_tokens_count - len(verb_tokens))]
                        label_ids = label_ids[:(self.max_seq_length - special_tokens_count - len(verb_tokens))]
                else:
                    special_tokens_count = 2
                    if self.sep_token_extra:
                        special_tokens_count += 1
                    if len(tokens) > self.max_seq_length - special_tokens_count:
                        if task.startswith("PARSING"):
                            skip_num += 1 # skip long sentence
                            continue
                        else:
                            tokens = tokens[:(self.max_seq_length - special_tokens_count)]
                            label_ids = label_ids[:(self.max_seq_length - special_tokens_count)]
                
                if task.startswith("PARSING"):
                    # parsing need new position
                    orig_to_tok_index = [x+1 for x in orig_to_tok_index] # +1 for the start [CLS]
                    new_head_ids = []
                    for x in head_ids:
                        if x == 0:
                            new_head_ids += [0] # ROOT will be overlapped with [CLS]
                        elif x == -100:
                            new_head_ids += [-100] # PAD token keep same , will be igonred during CrossEntropy
                        else:
                            new_head_ids += [orig_to_tok_index[x-1]] # PTB and UD parsing starts with position 1 , need to be 0
                    head_ids = new_head_ids # redirect 
                    
                tokens += [self.tokenizer.sep_token]
                label_ids += [-100]
                if self.sep_token_extra:
                    tokens += [self.tokenizer.sep_token]
                    label_ids += [-100]
                segment_ids = [0]*len(tokens)

                if task.startswith("PARSING"):
                    head_ids += [-100]
                    if self.sep_token_extra:
                        head_ids += [-100]
                

                tokens = [self.tokenizer.cls_token] + tokens
                
                label_ids = [-100] + label_ids
                segment_ids = [0] + segment_ids
                if task.startswith("PARSING"):
                    head_ids = [-100] + head_ids

                if task == "SRL":
                    if self.sep_token_extra:
                        # back a sep token
                        tokens = tokens[:-1]
                        label_ids = label_ids[:-1]
                        segment_ids = segment_ids[:-1]

                    tokens +=   verb_tokens + [self.tokenizer.sep_token]
                    label_ids +=   [-100]*(len(verb_tokens) + 1) 
                    segment_ids +=    [1]*(len(verb_tokens) + 1)

                    
                
                input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1]*len(input_ids)

                padding_length = self.max_seq_length - len(input_ids)

                input_ids += [0]*padding_length
                input_mask += [0]*padding_length
                segment_ids += [0]*padding_length
                label_ids += [-100]*padding_length
                if task.startswith("PARSING"):
                    head_ids += [-100]*padding_length

                assert len(input_ids) == self.max_seq_length
                assert len(input_mask) == self.max_seq_length
                assert len(segment_ids) == self.max_seq_length
                assert len(label_ids) == self.max_seq_length
                if task.startswith("PARSING"):
                    assert len(head_ids) == self.max_seq_length

                if ex_index < 5:
                    logger.info("*** {} Example ***".format(task))
                    logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
                    logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
                    logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
                    logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
                    logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))
                    if task.startswith("PARSING"):
                        logger.info("head_ids: %s", " ".join([str(x) for x in head_ids]))
                
                if not task.startswith("PARSING"):
                    head_ids = [0]*len(label_ids)
                    
                features.append([input_ids, input_mask, segment_ids,label_ids, head_ids])

            # instances = [[task_name, label_map, example] for example in examples]
            # with Pool(cpu_count()) as p:
            #     features = list(tqdm(p.imap(self.solve, instances), total=len(instances)))
            
            old_length = len(features)
            new_length = (old_length // batch_size) * batch_size
            features = features[:new_length]

            self.features_map["{}-{}".format(task, mode)] = features

        # logger.info("*** Statistics ***")
        # logger.info("*** max_len:{}  min_len:{} avg_len:{}***".format(max(cnt_counts), min(cnt_counts), sum(cnt_counts) / len(cnt_counts)))
        all_input_ids = torch.tensor([x[0] for x in features], dtype=torch.long)
        all_input_mask = torch.tensor([x[1] for x in features], dtype=torch.long)
        all_segment_ids = torch.tensor([x[2] for x in features], dtype=torch.long)
        all_label_ids = torch.tensor([x[3] for x in features], dtype=torch.long)
        all_head_ids = torch.tensor([x[4] for x in features], dtype=torch.long)    
        all_task_ids = torch.tensor([task_id for x in features], dtype=torch.long)
        
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_head_ids, all_task_ids)
        return features, dataset, task_id
    
    def load_MTDNN_dataset(self, batch_size, debug=False):
        all_data_sets = []
        all_input_ids = []
        all_input_mask = []
        all_segment_ids = []
        all_label_ids = []
        all_head_ids = []
        all_task_ids = []
        cache_path = os.path.join(self.datasets_dir, "joint_cache.pl")
        if os.path.exists(cache_path):
            print("Loading MTDNN datasets from cache {}".format(cache_path))
            all_dataset = torch.load(cache_path)
        else:
            for task in self.task_list:
                if debug:
                    features, dataset, task_id = self.load_single_dataset(task, batch_size, "debug")
                else:
                    features, dataset, task_id = self.load_single_dataset(task, batch_size, "train")
                all_input_ids += [x[0] for x in features]
                all_input_mask += [x[1] for x in features]
                all_segment_ids += [x[2] for x in features]
                all_label_ids += [x[3] for x in features]
                all_head_ids +=[x[4] for x in features]
                all_task_ids += [task_id for x in features]
            
            all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
            all_input_mask = torch.tensor(all_input_mask, dtype=torch.long)
            all_segment_ids = torch.tensor(all_segment_ids, dtype=torch.long)
            all_label_ids = torch.tensor(all_label_ids, dtype=torch.long)
            all_head_ids = torch.tensor(all_head_ids, dtype=torch.long)
            all_task_ids = torch.tensor(all_task_ids, dtype=torch.long)
            all_dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_head_ids, all_task_ids)
            print("saving MTNN datasets to cache {}".format(cache_path))
            torch.save(all_dataset, cache_path)

        all_dataset_sampler = RandomBatchSampler(all_dataset, batch_size)
        return all_dataset, all_dataset_sampler

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
                batch_t = features[cnt:cnt+self.mini_batch_size]
                features_batches.append(batch_t)
                cnt += self.mini_batch_size

            features_batches.append(features[cnt:])

            features_batches_list.append(features_batches)
        return features_batches_list

    def load_MTDNN_dataset_for_distribute(self, batch_size, debug=False):
        all_data_sets = []
        all_input_ids = []
        all_input_mask = []
        all_segment_ids = []
        all_label_ids = []
        all_head_ids = []
        all_task_ids = []
        for task in self.task_list:
            if debug:
                features, dataset, task_id = self.load_single_dataset(task, batch_size, "debug")
            else:
                features, dataset, task_id = self.load_single_dataset(task, batch_size, "train")

            all_input_ids += [x[0] for x in features]
            all_input_mask += [x[1] for x in features]
            all_segment_ids += [x[2] for x in features]
            all_label_ids += [x[3] for x in features]
            all_head_ids += [x[4] for x in features]
            all_task_ids += [task_id for x in features]

        all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
        all_input_mask = torch.tensor(all_input_mask, dtype=torch.long)
        all_segment_ids = torch.tensor(all_segment_ids, dtype=torch.long)
        all_label_ids = torch.tensor(all_label_ids, dtype=torch.long)
        all_head_ids = torch.tensor(all_head_ids, dtype=torch.long)
        all_task_ids = torch.tensor(all_task_ids, dtype=torch.long)
        all_dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_head_ids, all_task_ids)

        all_dataset_sampler = DistributedRandomBatchSampler(all_dataset, batch_size)
        return all_dataset, all_dataset_sampler


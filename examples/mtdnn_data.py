from __future__ import absolute_import, division, print_function

import logging
import os
from io import open
import yaml
from tqdm import *
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, ConcatDataset, BatchSampler

from multiprocessing import cpu_count, Pool

def load_MTDNN_dataset(data_folder, batch_size):
    # load each single task data
    all_data_sets = []
    for i range all_task:
        all_data_sets.append(load_single_dataset(batch_size))
    
    # !!!! drop tail, make sure the length is of each task equals n * batch_size
    # Example:  [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]] -> [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    
    # add task id in single_dataset TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, task_ids)
    all_dataset = ConcatDataset(all_data_sets)
    all_dataset_sampler = BatchSampler(all_dataset, batch_size)
    all_dataset_sampler = RandomSampler(all_dataset_sampler)
    return all_dataset, all_dataset_sampler
    
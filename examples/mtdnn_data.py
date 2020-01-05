from __future__ import absolute_import, division, print_function

import logging
import os
from io import open
import yaml
from tqdm import *
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, ConcatDataset, BatchSampler

from multiprocessing import cpu_count, Pool

class UniNLPMTDNNDataset():
    def __init__(self, data_folder, batch_size):
        # load each single task data
        all_data_sets = []
        for i range all_task:
            all_data_sets.append(load_single_dataset(batch_size))
        
        # !!!! drop tail, make sure the length is of each task equals n * batch_size
        # Example:  [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]] -> [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        all_dataset = ConcatDataset(all_data_sets)
        self.all_dataset = BatchSampler(all_dataset, batch_size)
        self.all_dataset = RandomSampler(self.all_dataset)

    def __iter__(self):
        next(self.all_dataset)
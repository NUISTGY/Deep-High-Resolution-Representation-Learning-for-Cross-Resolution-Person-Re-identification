from __future__ import absolute_import
from __future__ import division

from collections import defaultdict
import numpy as np
import copy
import random

import torch
from torch.utils.data.sampler import Sampler


class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.

    Args:
    - data_source (Dataset): dataset to sample from.
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """
    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source # train_all数据集:751个id
        self.batch_size = batch_size # 每个批次图片总数
        self.num_instances = num_instances # 每个id选取的图片张数
        self.num_pids_per_batch = self.batch_size // self.num_instances # 每个批次包含多少id
        self.index_dic = defaultdict(list) # 创建index字典
        for index, (_, _, pid ,_ ,_) in enumerate(self.data_source): # pid是list，pid=[0,1,2,3,...,750]作为index_dic的keys
            self.index_dic[pid].append(index) # keys对应的value为train_all中每个pid对应的图片序号
        self.pids = list(self.index_dic.keys())
        # 计算采样后数据集样本总数：self.length=3004=751*4 
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        # __iter__()函数中具体说明了图片采样的过程
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        # 返回采样处理后数据集包含的图片数：3004
        return self.length
'''
RPS network training on imagenet dataset
Copyright (c) Jathushan Rajasegaran, 2019
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import pickle
import torch
import pdb
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.autograd import gradcheck
import sys
import random


from rps_net import MultiHeadRPS_net
from learner import Learner
from util import *
from cifar_dataset import CIFAR100

from avalanche.benchmarks.utils import PathsDataset
from cgqa import continual_training_benchmark, fewshot_testing_benchmark

from cgqa_continual import main


class args:
    epochs = 100
    checkpoint = "/apdcephfs/share_1364275/lwd/RPSnet-experiments/results/cgqa/RPSnet-resnet-M8-lr1e-3"
    savepoint =  "/apdcephfs/share_1364275/lwd/RPSnet-experiments/models/cgqa/RPSnet-resnet-M8-lr1e-3"
    data = '/apdcephfs/share_1364275/lwd/datasets'
    return_task_id = False      # True for task-IL, False for class-IL
    # labels_data = "prepare/sysgqa_train.pkl"

    image_size = 128

    num_class = 100         # no use
    # for task-IL, should be 10, for class-IL, should be 100
    class_per_task = 10
    M = 8
    jump = 2
    rigidness_coff = 10
    dataset = "CGQA"
    num_train_task = 10     # only related to sess, for task-IL and class-IL, it is 10.
    num_test_task = 300     # with num_class together, use to define the classifier: (300 + 10) * [100]
    num_test_class = 10

    L = 9
    N = 1
    lr = 0.001
    train_batch = 100
    test_batch = 100
    workers = 10
    resume = False
    arch = "res-18"
    start_epoch = 0
    evaluate = False
    sess = 0
    test_case = 0
    schedule = [20, 40, 60, 80]
    gamma = 0.5


state = {key:value for key, value in args.__dict__.items() if not key.startswith('__') and not callable(key)}
print(state)

# Use CUDA
use_cuda = torch.cuda.is_available()

seed = random.randint(1, 10000)
random.seed(seed)
torch.manual_seed(seed)
if use_cuda:
    torch.cuda.manual_seed_all(seed)

    
if __name__ == '__main__':
    main(args)


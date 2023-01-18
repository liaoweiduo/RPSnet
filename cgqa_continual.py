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


class args:
    epochs = 100
    checkpoint = "../RPSnet-experiments/results/cgqa/RPSnet-cls"
    savepoint = "../RPSnet-experiments/models/cgqa/RPSnet-cls"
    data = '../datasets'
    return_task_id = False      # True for task-IL, False for class-IL
    # labels_data = "prepare/sysgqa_train.pkl"
    
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
    train_batch = 50
    test_batch = 50
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


def main(args):
    
    
    model = MultiHeadRPS_net(args).cuda()
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    
    
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    if not os.path.isdir(args.savepoint):
        mkdir_p(args.savepoint)


    start_sess = int(sys.argv[2])
    test_case = sys.argv[1]

    args.test_case = test_case

    if start_sess < args.num_train_task:     # continual train
        sess_offset = 0
        benchmark = continual_training_benchmark(
            n_experiences=args.num_train_task, return_task_id=args.return_task_id,
            seed=1234, shuffle=True,
            dataset_root=args.data,
            memory_size=1000,
        )
    else:
        if start_sess < args.num_train_task + 1 * args.num_test_task:      # sys
            sess_offset = args.num_train_task
            mode = 'sys'
        elif start_sess < args.num_train_task + 2 * args.num_test_task:      # pro
            sess_offset = args.num_train_task + 1 * args.num_test_task
            mode = 'pro'
        elif start_sess < args.num_train_task + 3 * args.num_test_task:      # sub
            sess_offset = args.num_train_task + 2 * args.num_test_task
            mode = 'sub'
        elif start_sess < args.num_train_task + 4 * args.num_test_task:      # non
            sess_offset = args.num_train_task + 3 * args.num_test_task
            mode = 'non'
        elif start_sess < args.num_train_task + 5 * args.num_test_task:      # noc
            sess_offset = args.num_train_task + 4 * args.num_test_task
            mode = 'noc'

        elif start_sess < args.num_train_task + 6 * args.num_test_task:  # sys    no freeze fe
            sess_offset = args.num_train_task + 5 * args.num_test_task
            mode = 'sys'
        elif start_sess < args.num_train_task + 7 * args.num_test_task:  # pro
            sess_offset = args.num_train_task + 6 * args.num_test_task
            mode = 'pro'
        elif start_sess < args.num_train_task + 8 * args.num_test_task:  # sub
            sess_offset = args.num_train_task + 7 * args.num_test_task
            mode = 'sub'
        elif start_sess < args.num_train_task + 9 * args.num_test_task:  # non
            sess_offset = args.num_train_task + 8 * args.num_test_task
            mode = 'non'
        elif start_sess < args.num_train_task + 10 * args.num_test_task:  # noc
            sess_offset = args.num_train_task + 9 * args.num_test_task
            mode = 'noc'
        else:
            raise Exception(f'sess error: {start_sess}.')

        task_offset = args.num_train_task if args.return_task_id else 1
        benchmark = fewshot_testing_benchmark(
            n_experiences=args.num_test_task, mode=mode,
            n_way=10, n_shot=10, n_val=5, n_query=10,
            task_offset=task_offset,
            seed=1234,
            dataset_root=args.data)

        
    for ses in range(start_sess, start_sess+1):

        ##############################  data loader #####################

        train_dataset = benchmark.train_stream[ses - sess_offset].dataset
        val_dataset = benchmark.test_stream[ses - sess_offset].dataset

        train_sampler = None

        trainloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.train_batch, shuffle=(train_sampler is None),
            # num_workers=args.workers,
            pin_memory=True, sampler=train_sampler)

        testloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.test_batch, shuffle=False,
            # num_workers=args.workers,
            pin_memory=True)
        ############################## data loader  ######################

        load_model_ses = ses-1
        if(ses==0):
            path = get_path(args.L,args.M,args.N)*0 
            path[:,0] = 1
            fixed_path = get_path(args.L,args.M,args.N)*0 
            train_path = path.copy()
            infer_path = path.copy()
        else:
            if ses >= args.num_train_task:   # in novel testing phase
                load_model_ses = args.num_train_task - 1      # load the model after continual training phase

            load_test_case = get_best_model(load_model_ses, args.checkpoint)
            print(f'get_best_model: {load_model_ses}, with test case: {load_test_case}')
            if(ses%args.jump==0) or ses >= args.num_train_task:   # get a new path             for all novel testing,
                fixed_path = np.load(args.checkpoint+"/fixed_path_"+str(load_model_ses)+"_"+str(load_test_case)+".npy")
                train_path = get_path(args.L,args.M,args.N)*0 
                path = get_path(args.L,args.M,args.N)
            else:
                if((ses//args.jump)*2==0):  # ses == 1
                    fixed_path = get_path(args.L,args.M,args.N)*0
                else:
                    load_test_case_x = get_best_model((ses//args.jump)*2-1, args.checkpoint)
                    print(f'get_best_model_: {(ses//args.jump)*2-1}, with test case: {load_test_case_x}')
                    fixed_path = np.load(args.checkpoint+"/fixed_path_"+str((ses//args.jump)*2-1)+"_"+str(load_test_case_x)+".npy")
                path = np.load(args.checkpoint+"/path_"+str(load_model_ses)+"_"+str(load_test_case)+".npy")
            
                train_path = get_path(args.L,args.M,args.N)*0 
            infer_path = get_path(args.L,args.M,args.N)*0 
            for j in range(args.L):
                for i in range(args.M):
                    if(fixed_path[j,i]==0 and path[j,i]==1):
                        train_path[j,i]=1
                    if(fixed_path[j,i]==1 or path[j,i]==1):
                        infer_path[j,i]=1
            
        np.save(args.checkpoint+"/path_"+str(ses)+"_"+str(test_case)+".npy", path)
        
        
        if(ses==0):
            fixed_path_x = path.copy()
        else:
            fixed_path_x = fixed_path.copy()
            for j in range(args.L):
                for i in range(args.M):
                    if(fixed_path_x[j,i]==0 and path[j,i]==1):
                        fixed_path_x[j,i]=1
        np.save(args.checkpoint+"/fixed_path_"+str(ses)+"_"+str(test_case)+".npy", fixed_path_x)
        
        
        
        print('Starting with session {:d}'.format(ses))
        print('test case : ' + str(test_case))
        print('#################################################################################')
        print("path\n",path)
        print("fixed_path\n",fixed_path)
        print("train_path\n", train_path)

        

        print('trn_instances len:', len(train_dataset.targets))
        print('val_instances len:', len(val_dataset.targets))
        
        
        
        args.sess=ses      
        if ses>0: 
            path_model=os.path.join(args.savepoint, 'session_'+str(load_model_ses)+'_'+str(load_test_case)+'_model_best.pth.tar')
            prev_best=torch.load(path_model)
            model.load_state_dict(prev_best['state_dict'])
            print(f'load model from {path_model}.')

            if ses >= args.num_train_task and ses < args.num_train_task + 5 * args.num_test_task:   # afterward also train fe
                model.freeze_feature_extractor()


        main_learner=Learner(model=model,args=args,trainloader=trainloader,
                             testloader=testloader,old_model=copy.deepcopy(model),
                             use_cuda=use_cuda, path=path, 
                             fixed_path=fixed_path, train_path=train_path, infer_path=infer_path)
        main_learner.learn()
        

        if(ses==0):
            fixed_path = path.copy()
        else:
            for j in range(args.L):
                for i in range(args.M):
                    if(fixed_path[j,i]==0 and path[j,i]==1):
                        fixed_path[j,i]=1
        np.save(args.checkpoint+"/fixed_path_"+str(ses)+"_"+str(test_case)+".npy", fixed_path)
        
        # best_model = get_best_model(ses, args.checkpoint)
        
        
    print('done with session {:d}'.format(ses))
    print('#################################################################################')
    while(1):
        if(is_all_done(ses, args.epochs, args.checkpoint)):
            break
        else:
            time.sleep(30)
            
    
if __name__ == '__main__':
    main(args)


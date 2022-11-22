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


from rps_net import RPS_net
from learner import Learner
from util import *
from cifar_dataset import CIFAR100

from avalanche.benchmarks.utils import PathsDataset


class args:
    epochs = 60
    checkpoint = "results/subgqa/RPSnet"
    savepoint = ""
    data = '../datasets/gqa/allImages/images'
    labels_data = "prepare/subgqa_color_train.pkl"
    
    num_class = 40
    class_per_task = 2
    M = 8
    jump = 2
    rigidness_coff = 10
    dataset = "SUBGQA"
   
    L = 9
    N = 1
    lr = 0.001
    train_batch = 16
    test_batch = 16
    workers = 8
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


def main():
    
    
    model = RPS_net(args).cuda() 
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    
    
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)
        
    if not os.path.isdir("models/subgqa/"+args.checkpoint.split("/")[-1]):
        mkdir_p("models/subgqa/"+args.checkpoint.split("/")[-1])
    args.savepoint = "models/subgqa/"+args.checkpoint.split("/")[-1]


    start_sess = int(sys.argv[2])
    test_case = sys.argv[1]
    
    args.test_case = test_case

    a=pickle.load(open(args.labels_data,'rb'))

        
    for ses in range(start_sess, start_sess+1):

        ##############################  data loader for sys gqa dataset #####################

        trn_instances = [a['trn_ind'][idx] for idx in a[ses]['curent']]
        val_instances = [a['tst_ind'][idx] for idx in a[ses]['test']]

        ex_instances = []
        if ses > 0 and ses < 10:        # for novel testing, do not use replay memory
            ex_instances = [a['trn_ind'][idx] for idx in a[ses-1]['exmp']]
            trn_instances.extend(ex_instances)

        # train_dataset=ImageFilelist(root=args.data, flist=trn_fnames,targets=trn_labs,transform= transforms.Compose([
        #         transforms.RandomResizedCrop(224),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         normalize,
        #     ]))
        #
        # val_dataset=ImageFilelist(root=args.data, flist=val_fnames,targets=val_labs,transform= transforms.Compose([
        #         transforms.Resize(230),
        #         transforms.CenterCrop(224),
        #         transforms.ToTensor(),
        #         normalize,
        #     ]))

        _image_size = (224, 224)
        _default_cgqa_train_transform = transforms.Compose(
            [
                transforms.Resize(_image_size),  # allow reshape but not equal scaling
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        _default_cgqa_eval_transform = transforms.Compose(
            [
                transforms.Resize(_image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        def target_transform(ses):
            '''-----------'''
            '''Train phase'''
            '''-----------'''

            train_classes_in_exp = [[16, 15], [17, 14], [1, 9], [0, 12], [6, 7], [5, 13], [2, 18], [11, 3], [8, 4],
                                    [10, 19]]
            train_classes_related = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17],
                                     [18, 19]]

            '''----------------'''
            '''Novel test phase'''
            '''----------------'''
            novel_test_classes_in_exp = np.array(
                [[15, 12], [1, 14], [10, 8], [3, 13], [0, 6], [4, 19], [9, 5], [16, 7], [11, 17], [18, 2]]) + 20
            # apply label offset to novel test classes
            novel_test_classes_related = [[20, 21], [20, 21], [20, 21], [20, 21], [20, 21], [20, 21], [20, 21],
                                          [20, 21], [20, 21], [20, 21]]

            def target_transform_(target):
                if ses < 10:  # Training phase
                    class_mapping = np.array([None for _ in range(np.max(train_classes_in_exp[:ses+1]) + 1)])
                    for s in range(ses+1):
                        class_mapping[train_classes_in_exp[s]] = train_classes_related[s]
                else:       # Novel testing phase
                    class_mapping = np.array([None for _ in range(max(novel_test_classes_in_exp[ses]) + 1)])
                    class_mapping[novel_test_classes_in_exp[ses]] = novel_test_classes_related[ses]

                return class_mapping[target]

            return target_transform_

        train_dataset = PathsDataset(
            root=args.data,
            files=trn_instances,
            transform=_default_cgqa_train_transform,
            target_transform=target_transform(ses)
        )
        val_dataset = PathsDataset(
            root=args.data,
            files=val_instances,
            transform=_default_cgqa_eval_transform,
            target_transform=target_transform(ses)
        )

        train_sampler = None

        trainloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.train_batch, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=False, sampler=train_sampler)

        testloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.test_batch, shuffle=False,
            num_workers=args.workers, pin_memory=False)
        ############################## data loader for imagenet based upon file names ######################

        load_model_ses = ses-1
        if(ses==0):
            path = get_path(args.L,args.M,args.N)*0 
            path[:,0] = 1
            fixed_path = get_path(args.L,args.M,args.N)*0 
            train_path = path.copy()
            infer_path = path.copy()
        else:
            if ses >= 10:   # in novel testing phase
                load_model_ses = 9      # load the model after continual training phase

            load_test_case = get_best_model(load_model_ses, args.checkpoint)
            print(f'get_best_model: {load_model_ses}, with test case: {load_test_case}')
            if(ses%args.jump==0) or ses >= 10:   # get a new path             only for continual training. for novel testing,
                fixed_path = np.load(args.checkpoint+"/fixed_path_"+str(load_model_ses)+"_"+str(load_test_case)+".npy")
                train_path = get_path(args.L,args.M,args.N)*0 
                path = get_path(args.L,args.M,args.N)
            else:
                if((ses//args.jump)*2==0):  # ses == 1
                    fixed_path = get_path(args.L,args.M,args.N)*0
                else:
                    load_model_ses_ = (ses//args.jump)*2-1

                    load_test_case_x = get_best_model(load_model_ses_, args.checkpoint)
                    print(f'get_best_model: {load_model_ses_}, with test case: {load_test_case_x}')
                    fixed_path = np.load(args.checkpoint+"/fixed_path_"+str(load_model_ses_)+"_"+str(load_test_case_x)+".npy")
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

        

        print('trn_instances len:', len(trn_instances))
        print('val_instances len:', len(val_instances))
        if(ses>0):
            print('ex_instances len:', len(ex_instances))
        
        
        
        args.sess=ses      
        if ses>0:
            path_model=os.path.join(args.savepoint, 'session_'+str(load_model_ses)+'_'+str(load_test_case)+'_model_best.pth.tar')
            prev_best=torch.load(path_model)
            model.load_state_dict(prev_best['state_dict'])
            print(f'load model from {path_model}.')


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
        
        best_model = get_best_model(ses, args.checkpoint)
        
        
    print('done with session {:d}'.format(ses))
    print('#################################################################################')
    while(1):
        if(is_all_done(ses, args.epochs, args.checkpoint)):
            break
        else:
            time.sleep(10)
            
    
if __name__ == '__main__':
    main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 10:51:09 2019
@author: munawar
"""
import torchvision.transforms as transforms
import itertools
import operator
from functools import reduce
import random
import numpy as np
import torchvision

import pickle

d=pickle.load(open('meta_CIFAR100.pkl','rb'))
ytrain=d['ytr'];ytest=d['yt']
# label of each sample: ndarray: {50000,}  ndarray: {10000,}

classes=np.arange(0,100)
inds_classes=[[np.where(ytrain==cl)] for cl in classes]


num_class = 100
task = 10
class_per_task = int(num_class/task)

def flatten_list(a): return np.concatenate(a).ravel()


def get_ind_sessions(ytrain,ytest):
    sessions=[]
    st=0;endd=class_per_task;
    for ii in range(task):
        sessions.append([np.arange(st,endd)])
        st=endd
        endd=st+class_per_task

    # sessions: classes in each task [0, 1,..., 9]; [10, 11,..., 19];...
    
    memb=2000       # number of exmp
    
    
    indices_final=dict()
    
    '''exm 包含了所有seen classes的平均的random train samples as a replay memory'''
    for session in range(len(sessions)):
        ind_curr=flatten_list([np.where(ytrain==c) for c in flatten_list(sessions[session])])
        # sample index in the classes of the current training task.  ndarray: {5000,}

        if session==0:
            ind_all=ind_curr
        else:
            ind_prev=indices_final[session-1]['exmp']
            ind_all=np.append(ind_curr,ind_prev)
            
        Ncl=1+sessions[session][0][-1]      # num of seen classes: 10 in 1st session, 20 in 2nd session.
        n_cl=int(memb/Ncl)                  # num of samples for each class: 200 in 1st session, 100 in 2nd session.
        ys=ytrain[np.array(ind_all)]
        # labels for all selected samples.
        exm=[]
        for ii in range(Ncl):
            ind_one_cl= ind_all[np.where(ys==ii)]
            # sample index for ii labels
            random.shuffle(ind_one_cl)
            exm.append(ind_one_cl[:n_cl])
            # randomly select n_cl samples
        exm=flatten_list(exm)
            
            
        
        all_test_classes=np.arange(0,class_per_task*(session+1)).astype(int)
        # for each task add 1000 samples for test, but increase to 2000 for the second, to 3000 for the third...
        session_test_ind=flatten_list([np.where(ytest==c) for c in all_test_classes])
        
        indices_final[session]={'curent': ind_curr, 'exmp': exm,'test':session_test_ind}
    return indices_final

indices_final=get_ind_sessions(ytrain,ytest)


pickle.dump(indices_final, open('cifar100_'+str(task)+'.pkl','wb'))

 

    


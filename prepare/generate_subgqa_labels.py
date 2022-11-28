#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 3 2022

@author: Weiduo Liao
"""
import os
import itertools
import operator
from functools import reduce
import numpy as np
import torchvision
import pickle
import json

mode = 'novel_test'      # option: [train, novel_test, non_novel_test]

data_json_dir = '../../datasets/gqa/sub_gqa_json/'

train_num_class = 20
novel_test_num_class = 20
num_class = train_num_class + novel_test_num_class  # 40

train_task = 10
novel_test_task = 600
non_novel_test_task = 600

if mode == 'train':
    num_task = train_task
elif mode == 'novel_test':
    num_task = novel_test_task
else:
    num_task = non_novel_test_task

class_per_task = 2

num_samples_each_label = 5      # few-shot support set size for each label,  total 10 samples

if mode == 'train':
    '''Load train information'''
    with open(os.path.join(data_json_dir, 'attriJson', 'attri_comb_train.json'), 'r') as f:
        train_img_info = json.load(f)
    with open(os.path.join(data_json_dir, 'attriJson', 'attri_comb_test.json'), 'r') as f:
        test_img_info = json.load(f)
elif mode == 'novel_test':
    '''Load Novel test information'''
    with open(os.path.join(data_json_dir, 'attriJson', 'novel_attri_comb_train.json'), 'r') as f:
        train_img_info = json.load(f)
    with open(os.path.join(data_json_dir, 'attriJson', 'novel_attri_comb_test.json'), 'r') as f:
        test_img_info = json.load(f)
elif mode == 'non_novel_test':
    '''Load train information'''
    with open(os.path.join(data_json_dir, 'attriJson', 'attri_comb_train.json'), 'r') as f:
        train_img_info = json.load(f)
    with open(os.path.join(data_json_dir, 'attriJson', 'attri_comb_test.json'), 'r') as f:
        test_img_info = json.load(f)
else:
    raise Exception(f'Un-implemented mode: {mode}.')

'''preprocess labels to integers'''
# for continual training
label_set = sorted(list(set([item['label'][0] for item in test_img_info])))
# 0-19
map_tuple_label_to_int = dict((item, idx) for idx, item in enumerate(label_set))    # only 0-19
map_int_label_to_tuple = dict((idx, item) for idx, item in enumerate(label_set))    # only 0-19


for item in train_img_info:
    item['image'] = f"{item['image']}.jpg"
    item['label'] = map_tuple_label_to_int[item['label'][0]]
for item in test_img_info:
    item['image'] = f"{item['image']}.jpg"
    item['label'] = map_tuple_label_to_int[item['label'][0]]

'''if in train or non_novel_test mode, build-in seed is used to select specific train images'''
if mode in ['train', 'non_novel_test']:
    imgs_each_label = dict()
    for item in train_img_info:
        label = item['label']
        if label in imgs_each_label:
            imgs_each_label[label].append(item)
        else:
            imgs_each_label[label] = [item]

    build_in_seed = 1234
    build_in_rng = np.random.RandomState(seed=build_in_seed)

    selected_train_images = []
    selected_test_images = []
    for label, imgs in imgs_each_label.items():
        # random permutation
        idxs_perm = build_in_rng.permutation(np.arange(len(imgs)))
        num_non_novel_train, num_non_novel_test = 50, 50
        if mode == 'non_novel_test':  # first 50+50 for non_novel_testing
            selected_idxs_train = idxs_perm[: num_non_novel_train]
            selected_idxs_test = idxs_perm[num_non_novel_train: num_non_novel_train + num_non_novel_test]
            for idx in selected_idxs_train:
                selected_train_images.append(imgs[idx])
            for idx in selected_idxs_test:
                selected_test_images.append(imgs[idx])
        else:  # all others for train
            selected_idxs = idxs_perm[num_non_novel_train + num_non_novel_test:]
            for idx in selected_idxs:
                selected_train_images.append(imgs[idx])

    train_img_info = selected_train_images
    if mode == 'non_novel_test':
        test_img_info = selected_test_images

ytrain = np.array([item['label'] for item in train_img_info])   # from 0 to 19 or 20 to 39
ytest = np.array([item['label'] for item in test_img_info])     # from 0 to 19 or 20 to 39

train_list = []
for item in train_img_info:
    bbn_name = 'boundingbox' if 'boundingbox' in item.keys() else 'boundingBox'
    instance_tuple = (item['image'], item['label'], item[bbn_name])
    train_list.append(instance_tuple)
test_list = []
for item in test_img_info:
    bbn_name = 'boundingbox' if 'boundingbox' in item.keys() else 'boundingBox'
    instance_tuple = (item['image'], item['label'], item[bbn_name])
    test_list.append(instance_tuple)
# [('2325499C73236.jpg', 0, [2, 4, 335, 368]), ('2369086C73237.jpg', 0, [2, 4, 335, 368]),...


inds_classes = [[np.where(ytrain==cl)] for cl in np.arange(0, train_num_class)]             # 0-19

# '''Only top few classes are considered'''
# top=np.where(np.array(ytrain)<num_class);ytrain=np.array(ytrain)[top]   # ;trn_nms=np.array(trn_nms)[top]
# top=np.where(np.array(ytest)<num_class);ytest=np.array(ytest)[top]      # ;tst_nms=np.array(tst_nms)[top]


# classes=np.arange(0,num_class)
# seed = 1234
# rng = np.random.RandomState(seed)
# rng.shuffle(classes[:20])
# # random.shuffle(classes[:20])
# # only train classes are needed to be shuffled, the last 5 (novel test) do not.


def flatten_list(a):
    return np.concatenate(a).ravel()


def get_ind_sessions(ytrain, ytest):
    # sessions = []
    # st = 0;
    # endd = class_per_task;
    # for ii in range(task):
    #     sessions.append([np.arange(st, endd)])
    #     st = endd
    #     endd = st + class_per_task

    indices_final = dict()

    '''-----------'''
    '''Train phase'''
    '''-----------'''
    train_classes_in_exp = [[16, 15], [17, 14], [1, 9], [0, 12], [6, 7], [5, 13], [2, 18], [11, 3], [8, 4], [10, 19]]
    train_classes_related = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19]]

    sessions = [[classes] for classes in train_classes_in_exp]

    '''----------------'''
    '''Novel test phase'''
    '''----------------'''
    seed = 1234
    rng = np.random.RandomState(seed)
    novel_test_classes = np.arange(novel_test_num_class)  # [0-19]
    classes_in_exp = []
    for task_idx in range(novel_test_task):
        task_classes = rng.choice(novel_test_classes, class_per_task, replace=False)
        classes_in_exp.append(task_classes)

        if mode == 'novel_test':
            '''select num_samples_each_label*num_ways_each_task images for each exp'''
            ind_curr = []
            for cls_idx in task_classes:
                ind_curr.append(rng.choice(np.where(ytrain == int(cls_idx))[0], num_samples_each_label, replace=False))
            ind_curr = np.concatenate(ind_curr)

            exm = []

            session_test_ind = flatten_list([np.where(ytest == c)[0] for c in flatten_list([task_classes])])

            indices_final[task_idx + train_task] = {'curent': ind_curr, 'exmp': exm, 'test': session_test_ind}
            # from 10-609

    novel_test_classes_in_exp = np.stack(classes_in_exp)
    novel_test_classes_related = np.array([[20, 21] for _ in range(novel_test_task)])

    sessions.extend([[classes] for classes in novel_test_classes_in_exp])

    '''--------------------'''
    '''Non Novel test phase'''
    '''--------------------'''
    seed = 1234
    rng = np.random.RandomState(seed)
    non_novel_test_classes = np.arange(train_num_class)  # [0-19]
    classes_in_exp = []
    for task_idx in range(non_novel_test_task):
        task_classes = rng.choice(non_novel_test_classes, class_per_task, replace=False)
        classes_in_exp.append(task_classes)

        if mode == 'non_novel_test':
            '''select num_samples_each_label*num_ways_each_task images for each exp'''
            ind_curr = []
            for cls_idx in task_classes:
                ind_curr.append(rng.choice(np.where(ytrain == int(cls_idx))[0], num_samples_each_label, replace=False))
            ind_curr = np.concatenate(ind_curr)

            exm = []

            session_test_ind = flatten_list([np.where(ytest == c)[0] for c in flatten_list([task_classes])])

            indices_final[task_idx + novel_test_task + train_task] = {'curent': ind_curr, 'exmp': exm, 'test': session_test_ind}
            # from 610-1209

    non_novel_test_classes_in_exp = np.stack(classes_in_exp)
    non_novel_test_classes_related = np.array([[20, 21] for _ in range(non_novel_test_task)])

    sessions.extend([[classes] for classes in non_novel_test_classes_in_exp])

    indices_final['sessions'] = sessions

    if mode == 'train':
        memb = 5000

        seed = 1234
        rng = np.random.RandomState(seed)

        for session in range(train_task):

            ind_curr = flatten_list([np.where(ytrain == c)[0] for c in flatten_list(sessions[session])])

            if session == 0 or session >= train_task:       # first and novel testing do not use mem
                ind_all = ind_curr
            else:
                ind_prev = indices_final[session - 1]['exmp']
                ind_all = np.append(ind_curr, ind_prev)

            exm = []
            if session < train_task:        # replay memory is only for train phase
                '''random sample some instances from each seen class'''
                seen_classes = flatten_list(sessions[:session+1])
                Ncl = len(seen_classes)
                n_cl = int(memb / Ncl)
                ys = ytrain[np.array(ind_all)]
                for ii in seen_classes:
                    ind_one_cl = ind_all[np.where(ys == ii)]
                    rng.shuffle(ind_one_cl)
                    exm.append(ind_one_cl[:n_cl])
                exm = flatten_list(exm)

            if session >= train_task:       # novel test phase only test on self.
                all_test_classes = sessions[session][0]
                # np.arange(class_per_task * session, class_per_task * (session + 1)).astype(int)
                session_test_ind = flatten_list([np.where(ytest == c)[0] for c in all_test_classes])
            else:
                all_test_classes = flatten_list(sessions[:session+1])
                # np.arange(0, class_per_task * (session + 1)).astype(int)
                session_test_ind = flatten_list([np.where(ytest == c)[0] for c in all_test_classes])

            indices_final[session] = {'curent': ind_curr, 'exmp': exm, 'test': session_test_ind}
    return indices_final


indices_final=get_ind_sessions(ytrain.flatten(),ytest.flatten())

indices_final['trn_ind']=train_list
indices_final['tst_ind']=test_list
indices_final['ytrain']=ytrain
indices_final['ytest']=ytest




if mode == 'train':
    pickle.dump(indices_final, open('subgqa_color_train.pkl', 'wb'))
elif mode == 'novel_test':
    pickle.dump(indices_final, open('subgqa_color_novel_test.pkl', 'wb'))
elif mode == 'non_novel_test':
    pickle.dump(indices_final, open('subgqa_color_non_novel_test.pkl','wb'))



 

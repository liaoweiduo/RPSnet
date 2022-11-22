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

mode = 'train'      # option: [train, novel_test, non_novel_test]

data_json_dir = '../../datasets/gqa/sys_gqa_json/'

seed = 1234
rng = np.random.RandomState(seed)

train_num_class = 20
novel_test_num_class = 5
num_class = train_num_class + novel_test_num_class  # 25

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

'''-----------'''
'''Train phase'''
'''-----------'''

train_classes_in_exp = [[16, 15], [17, 14], [1, 9], [0, 12], [6, 7], [5, 13], [2, 18], [11, 3], [8, 4], [10, 19]]
train_classes_related = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19]]

'''----------------'''
'''Novel test phase'''
'''----------------'''
# TBD to random generate
novel_test_classes_in_exp = [[24, 20], [23, 22], [22, 20], [21, 24], [21, 20], [22, 23], [24, 20], [24, 21],
                             [20, 21], [24, 22]]
novel_test_classes_related = [[20, 21] for _ in range(novel_test_task)]

'''--------------------'''
'''Non Novel test phase'''
'''--------------------'''
# TBD to random generate
non_novel_test_classes_in_exp = [[16, 15], [17, 14], [1, 9], [0, 12], [6, 7], [5, 13], [2, 18],
                                 [11, 3], [8, 4], [10, 19]]
non_novel_test_classes_related = [[20, 21] for _ in range(novel_test_task)]

if mode == 'train':
    '''Load train information'''
    with open(os.path.join(data_json_dir, 'comb_train.json'), 'r') as f:
        train_img_info = json.load(f)
    with open(os.path.join(data_json_dir, 'comb_test.json'), 'r') as f:
        test_img_info = json.load(f)
    label_offset = 0
elif mode == 'novel_test':
    '''Load Novel test information'''
    with open(os.path.join(data_json_dir, 'novel_comb_train.json'), 'r') as f:
        train_img_info = json.load(f)
    with open(os.path.join(data_json_dir, 'novel_comb_test.json'), 'r') as f:
        test_img_info = json.load(f)
    label_offset = train_num_class      # 20
elif mode == 'non_novel_test':
    '''Load train information'''
    with open(os.path.join(data_json_dir, 'comb_train.json'), 'r') as f:
        train_img_info = json.load(f)
    with open(os.path.join(data_json_dir, 'comb_test.json'), 'r') as f:
        test_img_info = json.load(f)
    label_offset = 0
else:
    raise Exception(f'Un-implemented mode: {mode}.')

'''preprocess labels to integers'''
label_set = sorted(list(set([tuple(sorted(item['label'])) for item in test_img_info])))
# 0-19 or 0-4

# [('building', 'sign'), ('building', 'sky'), ('building', 'window'), ('car', 'sign'), ('car', 'window'),
# ('grass', 'shirt'), ('grass', 'sky'), ('grass', 'tree'), ('hair', 'shirt'), ('hair', 'wall'), ('shirt', 'sign'),
# ('shirt', 'tree'), ('shirt', 'wall'), ('sign', 'sky'), ('sign', 'tree'), ('sign', 'wall'), ('sign', 'window'),
# ('sky', 'tree'), ('sky', 'window'), ('wall', 'window'),
# :test ('building', 'hair'), ('car', 'sky'), ('grass', 'sign'), ('shirt', 'window'), ('tree', 'wall')]
map_tuple_label_to_int = dict((item, idx + label_offset) for idx, item in enumerate(label_set))
map_int_label_to_tuple = dict((idx + label_offset, item) for idx, item in enumerate(label_set))


for item in train_img_info:
    item['image'] = f"{item['image']}.jpg"
    item['label'] = map_tuple_label_to_int[tuple(sorted(item['label']))]
for item in test_img_info:
    item['image'] = f"{item['image']}.jpg"
    item['label'] = map_tuple_label_to_int[tuple(sorted(item['label']))]

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

ytrain = np.array([item['label'] for item in train_img_info])   # from 0 to 19 or 20 to 24
ytest = np.array([item['label'] for item in test_img_info])     # from 0 to 19 or 20 to 24

train_list = []
for item in train_img_info:
    instance_tuple = (item['image'], item['label'], item['boundingBox'])
    train_list.append(instance_tuple)
test_list = []
for item in test_img_info:
    instance_tuple = (item['image'], item['label'], item['boundingBox'])
    test_list.append(instance_tuple)
# [('2325499C73236.jpg', 0, [2, 4, 335, 368]), ('2369086C73237.jpg', 0, [2, 4, 335, 368]),...


if mode in ['train', 'non_novel_test']:
    inds_classes = [[np.where(ytrain==cl)] for cl in np.arange(0, train_num_class)]             # 0-19
else:
    inds_classes = [[np.where(ytrain==cl)] for cl in np.arange(train_num_class, num_class)]     # 20-24

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

    sessions = [[classes] for classes in train_classes_in_exp]
    sessions.extend([[classes] for classes in novel_test_classes_in_exp])
    sessions.extend([[classes] for classes in non_novel_test_classes_in_exp])
    # [[[16, 15]], [[17, 14]],...]
    if mode == 'train':
        session_id_offset = 0
    elif mode == 'novel_test':
        session_id_offset = train_task  # 10
    else:
        session_id_offset = train_task + novel_test_task    # 10+600

    memb = 5000

    indices_final = dict()

    for session in range(session_id_offset, session_id_offset + num_task):

        ind_curr = flatten_list([np.where(ytrain == c)[0] for c in flatten_list(sessions[session])])


        # TBD, for few-shot testing, only sample some train samples for each class.


        if session == 0 or session >= train_task:       # first and all testing do not use mem
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
    pickle.dump(indices_final, open('sysgqa_train.pkl','wb'))
elif mode == 'novel_test':
    pickle.dump(indices_final, open('sysgqa_novel_test.pkl','wb'))
elif mode == 'non_novel_test':
    pickle.dump(indices_final, open('sysgqa_non_novel_test.pkl','wb'))

 

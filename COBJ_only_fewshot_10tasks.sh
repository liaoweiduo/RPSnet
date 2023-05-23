#!/bin/bash

## continual 0...2; sys 3..302; pro 303...602; non 903..1202; noc 1203..1502
#
##  $1 is lr, $2 is cuda
## bash COBJ.sh 1e-3
#
#for i_n in {3..602}   # for fewshot, only once, since use all trained modules when doing inference.
#do
#  CUDA_VISIBLE_DEVICES=$2 python3 cobj_continual_3tasks.py 0 $i_n $1
#done
#
#for i_n in {903..1502}   # for fewshot, only once, since use all trained modules when doing inference.
#do
#  CUDA_VISIBLE_DEVICES=$2 python3 cobj_continual_3tasks.py 0 $i_n $1
#done






### 10-tasks 3-way
## continual 0...9; sys 10..309; pro 310...609; non 910..1209; noc 1210..1509
#
##  $1 is lr, $2 is cuda
## bash COBJ.sh 1e-3

for i_n in {10..609}   # for fewshot, only once, since use all trained modules when doing inference.
do
  CUDA_VISIBLE_DEVICES=$2 python3 cobj_continual_10tasks.py 0 $i_n $1
done

for i_n in {910..1509}   # for fewshot, only once, since use all trained modules when doing inference.
do
  CUDA_VISIBLE_DEVICES=$2 python3 cobj_continual_10tasks.py 0 $i_n $1
done


## 5-tasks 6-way
# continual 0...4; sys 5..304; pro 305...604; non 905..1204; noc 1205..1504

#  $1 is lr, $2 is cuda
# bash COBJ.sh 1e-3

#for i_n in {5..604}   # for fewshot, only once, since use all trained modules when doing inference.
#do
#  CUDA_VISIBLE_DEVICES=$2 python3 cobj_continual_5tasks.py 0 $i_n $1
#done
#
#for i_n in {905..1504}   # for fewshot, only once, since use all trained modules when doing inference.
#do
#  CUDA_VISIBLE_DEVICES=$2 python3 cobj_continual_5tasks.py 0 $i_n $1
#done

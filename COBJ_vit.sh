#!/bin/bash

## continual 0...2; sys 3..302; pro 303...602; non 903..1202; noc 1203..1502
#
##  $1 is lr, $2 is cuda
## bash COBJ_vit.sh 1e-4

CUDA_VISIBLE_DEVICES=$2 python3 cobj_continual.py 0 0 $1

for i in {0..7}
do
  CUDA_VISIBLE_DEVICES=$2 python3 cobj_continual.py $i 1 $1
done

for i in {0..7}
do
  CUDA_VISIBLE_DEVICES=$2 python3 cobj_continual.py $i 2 $1
done

for i_n in {3..602}   # for fewshot, only once, since use all trained modules when doing inference.
do
  CUDA_VISIBLE_DEVICES=$2 python3 cobj_continual_3tasks.py 0 $i_n $1
done

for i_n in {903..1502}   # for fewshot, only once, since use all trained modules when doing inference.
do
  CUDA_VISIBLE_DEVICES=$2 python3 cobj_continual_3tasks.py 0 $i_n $1
done


#!/bin/bash

## continual 0...2; sys 3..302; pro 303...602; non 903..1202; noc 1203..1502
#
##  $1 is lr, $2 is cuda
## bash COBJ_vit.sh 1e-4 0

#CUDA_VISIBLE_DEVICES=$2 python3 cobj_continual_vit.py 0 0 $1

#for i in {7..7}
#do
#  CUDA_VISIBLE_DEVICES=$2 python3 cobj_continual_vit.py $i 1 $1
#done

for i in {3..7}
do
  CUDA_VISIBLE_DEVICES=$2 python3 cobj_continual_vit.py $i 2 $1
done

# only 100 few shot

for i_n in {3..102}   # for fewshot, only once, since use all trained modules when doing inference.
do
  CUDA_VISIBLE_DEVICES=$2 python3 cobj_continual_vit.py 0 $i_n $1
done

for i_n in {303..402}   # for fewshot, only once, since use all trained modules when doing inference.
do
  CUDA_VISIBLE_DEVICES=$2 python3 cobj_continual_vit.py 0 $i_n $1
done

for i_n in {903..1002}   # for fewshot, only once, since use all trained modules when doing inference.
do
  CUDA_VISIBLE_DEVICES=$2 python3 cobj_continual_vit.py 0 $i_n $1
done

for i_n in {1203..1302}   # for fewshot, only once, since use all trained modules when doing inference.
do
  CUDA_VISIBLE_DEVICES=$2 python3 cobj_continual_vit.py 0 $i_n $1
done



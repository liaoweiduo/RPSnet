#!/bin/bash

## continual 0...2; sys 3..302; pro 303...602; non 903..1202; noc 1203..1502
#
##  $1 is lr, $2 is cuda
## bash COBJ.sh 1e-3
#
##CUDA_VISIBLE_DEVICES=$2 python3 cobj_continual.py 0 0 $1
#
#for i in {0..7}
#do
#  CUDA_VISIBLE_DEVICES=$2 python3 cobj_continual.py $i 1 $1
#done
#
#for i in {0..7}
#do
#  CUDA_VISIBLE_DEVICES=$2 python3 cobj_continual.py $i 2 $1
#done
#
#for i_n in {3..602}   # for fewshot, only once, since use all trained modules when doing inference.
#do
#  CUDA_VISIBLE_DEVICES=$2 python3 cobj_continual.py 0 $i_n $1
#done
#
#for i_n in {903..1502}   # for fewshot, only once, since use all trained modules when doing inference.
#do
#  CUDA_VISIBLE_DEVICES=$2 python3 cobj_continual.py 0 $i_n $1
#done






## 10-tasks 3-way
# continual 0...9; sys 10..309; pro 310...609; non 910..1209; noc 1210..1509

#  $1 is lr, $2 is cuda
# bash COBJ.sh 1e-3

CUDA_VISIBLE_DEVICES=$2 python3 cobj_continual.py 0 0 $1

CUDA_VISIBLE_DEVICES=$2 python3 cobj_continual.py 0 1 $1

for i in {0..7}
do
  CUDA_VISIBLE_DEVICES=$2 python3 cobj_continual.py $i 2 $1
done

CUDA_VISIBLE_DEVICES=$2 python3 cobj_continual.py 0 3 $1

for i in {0..7}
do
  CUDA_VISIBLE_DEVICES=$2 python3 cobj_continual.py $i 4 $1
done

CUDA_VISIBLE_DEVICES=$2 python3 cobj_continual.py 0 5 $1

for i in {0..7}
do
  CUDA_VISIBLE_DEVICES=$2 python3 cobj_continual.py $i 6 $1
done

CUDA_VISIBLE_DEVICES=$2 python3 cobj_continual.py 0 7 $1

for i in {0..7}
do
  CUDA_VISIBLE_DEVICES=$2 python3 cobj_continual.py $i 8 $1
done

CUDA_VISIBLE_DEVICES=$2 python3 cobj_continual.py 0 9 $1

for i_n in {10..609}   # for fewshot, only once, since use all trained modules when doing inference.
do
  CUDA_VISIBLE_DEVICES=$2 python3 cobj_continual.py 0 $i_n $1
done

for i_n in {910..1509}   # for fewshot, only once, since use all trained modules when doing inference.
do
  CUDA_VISIBLE_DEVICES=$2 python3 cobj_continual.py 0 $i_n $1
done

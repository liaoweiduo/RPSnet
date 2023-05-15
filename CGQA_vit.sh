#!/bin/bash

# continual 0...9; sys 10..309; pro 310...609; sub 610..909; non 910..1209; noc 1210..1509

#CUDA_VISIBLE_DEVICES=0 python3 cgqa_continual_vit.py 0 0
#CUDA_VISIBLE_DEVICES=0 python3 cgqa_continual_vit.py 0 1

#for i in {0..7}
#do
#  CUDA_VISIBLE_DEVICES=0 python3 cgqa_continual_vit.py $i 2
#done

#CUDA_VISIBLE_DEVICES=0 python3 cgqa_continual_vit.py 0 3

for i in {4..7}
do
  CUDA_VISIBLE_DEVICES=0 python3 cgqa_continual_vit.py $i 4
done

CUDA_VISIBLE_DEVICES=0 python3 cgqa_continual_vit.py 0 5

for i in {0..7}
do
  CUDA_VISIBLE_DEVICES=0 python3 cgqa_continual_vit.py $i 6
done

CUDA_VISIBLE_DEVICES=0 python3 cgqa_continual_vit.py 0 7

for i in {0..7}
do
  CUDA_VISIBLE_DEVICES=0 python3 cgqa_continual_vit.py $i 8
done

CUDA_VISIBLE_DEVICES=0 python3 cgqa_continual_vit.py 0 9

for i_n in {10..1509}   # for fewshot, only once, since use all trained modules when doing inference.
do
  CUDA_VISIBLE_DEVICES=0 python3 cgqa_continual_vit.py 0 $i_n
done

# for only 2 GPUs
#for i_n in {4..1509}
#do
#  CUDA_VISIBLE_DEVICES=0 python3 cgqa_continual_vit.py 0 $i_n &
#  CUDA_VISIBLE_DEVICES=1 python3 cgqa_continual_vit.py 1 $i_n
#  sleep 20
#
#  CUDA_VISIBLE_DEVICES=0 python3 cgqa_continual_vit.py 2 $i_n &
#  CUDA_VISIBLE_DEVICES=1 python3 cgqa_continual_vit.py 3 $i_n
#  sleep 20
#
#  CUDA_VISIBLE_DEVICES=0 python3 cgqa_continual_vit.py 4 $i_n &
#  CUDA_VISIBLE_DEVICES=1 python3 cgqa_continual_vit.py 5 $i_n
#  sleep 20
#
#  CUDA_VISIBLE_DEVICES=0 python3 cgqa_continual_vit.py 6 $i_n &
#  CUDA_VISIBLE_DEVICES=1 python3 cgqa_continual_vit.py 7 $i_n
#  sleep 20
#
#done

# for only 4 GPUs
#for i_n in {4..1509}
#do
#  CUDA_VISIBLE_DEVICES=0 python3 cgqa_continual_vit.py 0 $i_n &
#  CUDA_VISIBLE_DEVICES=1 python3 cgqa_continual_vit.py 1 $i_n &
#  CUDA_VISIBLE_DEVICES=2 python3 cgqa_continual_vit.py 2 $i_n &
#  CUDA_VISIBLE_DEVICES=3 python3 cgqa_continual_vit.py 3 $i_n
#  sleep 20
#
#  CUDA_VISIBLE_DEVICES=0 python3 cgqa_continual_vit.py 4 $i_n &
#  CUDA_VISIBLE_DEVICES=1 python3 cgqa_continual_vit.py 5 $i_n &
#  CUDA_VISIBLE_DEVICES=2 python3 cgqa_continual_vit.py 6 $i_n &
#  CUDA_VISIBLE_DEVICES=3 python3 cgqa_continual_vit.py 7 $i_n
#  sleep 20
#
#done

# for only 8 GPUs
#for i_n in {0..1509}
#do
#  CUDA_VISIBLE_DEVICES=0 python3 cgqa_continual_vit.py 0 $i_n &
#  CUDA_VISIBLE_DEVICES=1 python3 cgqa_continual_vit.py 1 $i_n &
#  CUDA_VISIBLE_DEVICES=2 python3 cgqa_continual_vit.py 2 $i_n &
#  CUDA_VISIBLE_DEVICES=3 python3 cgqa_continual_vit.py 3 $i_n &
#  CUDA_VISIBLE_DEVICES=4 python3 cgqa_continual_vit.py 4 $i_n &
#  CUDA_VISIBLE_DEVICES=5 python3 cgqa_continual_vit.py 5 $i_n &
#  CUDA_VISIBLE_DEVICES=6 python3 cgqa_continual_vit.py 6 $i_n &
#  CUDA_VISIBLE_DEVICES=7 python3 cgqa_continual_vit.py 7 $i_n
#  sleep 20
#
#done
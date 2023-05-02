#!/bin/bash

# continual 0...9; sys 10..309; pro 310...609; sub 610..909; non 910..1209; noc 1210..1509

CUDA_VISIBLE_DEVICES=0 python3 cgqa_continual_vit.py 8 4
CUDA_VISIBLE_DEVICES=0 python3 cgqa_continual_vit.py 9 4
CUDA_VISIBLE_DEVICES=0 python3 cgqa_continual_vit.py 10 4
CUDA_VISIBLE_DEVICES=0 python3 cgqa_continual_vit.py 11 4

# for only 1 GPU
for i_n in {5..1509}
do
  CUDA_VISIBLE_DEVICES=0 python3 cgqa_continual_vit.py 0 $i_n
  CUDA_VISIBLE_DEVICES=0 python3 cgqa_continual_vit.py 1 $i_n
  CUDA_VISIBLE_DEVICES=0 python3 cgqa_continual_vit.py 2 $i_n
  CUDA_VISIBLE_DEVICES=0 python3 cgqa_continual_vit.py 3 $i_n
  CUDA_VISIBLE_DEVICES=0 python3 cgqa_continual_vit.py 4 $i_n
  CUDA_VISIBLE_DEVICES=0 python3 cgqa_continual_vit.py 5 $i_n
  CUDA_VISIBLE_DEVICES=0 python3 cgqa_continual_vit.py 6 $i_n
  CUDA_VISIBLE_DEVICES=0 python3 cgqa_continual_vit.py 7 $i_n
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
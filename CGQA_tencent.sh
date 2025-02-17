#!/bin/bash

# continual 0...9; sys 10..309; pro 310...609; sub 610..909; non 910..1209; noc 1210..1509

CUDA_VISIBLE_DEVICES=0 python3 cgqa_continual_tencent.py 4 0 &
CUDA_VISIBLE_DEVICES=1 python3 cgqa_continual_tencent.py 5 0 &
CUDA_VISIBLE_DEVICES=2 python3 cgqa_continual_tencent.py 6 0 &
CUDA_VISIBLE_DEVICES=3 python3 cgqa_continual_tencent.py 7 0
sleep 20

for i_n in {1..1509}
do
  CUDA_VISIBLE_DEVICES=0 python3 cgqa_continual_tencent.py 0 $i_n &
  CUDA_VISIBLE_DEVICES=1 python3 cgqa_continual_tencent.py 1 $i_n &
  CUDA_VISIBLE_DEVICES=2 python3 cgqa_continual_tencent.py 2 $i_n &
  CUDA_VISIBLE_DEVICES=3 python3 cgqa_continual_tencent.py 3 $i_n
  sleep 20

  CUDA_VISIBLE_DEVICES=0 python3 cgqa_continual_tencent.py 4 $i_n &
  CUDA_VISIBLE_DEVICES=1 python3 cgqa_continual_tencent.py 5 $i_n &
  CUDA_VISIBLE_DEVICES=2 python3 cgqa_continual_tencent.py 6 $i_n &
  CUDA_VISIBLE_DEVICES=3 python3 cgqa_continual_tencent.py 7 $i_n
  sleep 20

done

#!/bin/bash


# continual training phase


CUDA_VISIBLE_DEVICES=0 python3 split_sub_gqa.py 0 0 &
CUDA_VISIBLE_DEVICES=1 python3 split_sub_gqa.py 1 0 &
CUDA_VISIBLE_DEVICES=2 python3 split_sub_gqa.py 2 0 &
CUDA_VISIBLE_DEVICES=3 python3 split_sub_gqa.py 3 0 &
CUDA_VISIBLE_DEVICES=4 python3 split_sub_gqa.py 4 0 &
CUDA_VISIBLE_DEVICES=5 python3 split_sub_gqa.py 5 0 &
CUDA_VISIBLE_DEVICES=6 python3 split_sub_gqa.py 6 0 &
CUDA_VISIBLE_DEVICES=7 python3 split_sub_gqa.py 7 0


sleep 20
CUDA_VISIBLE_DEVICES=0 python3 split_sub_gqa.py 0 1 &
CUDA_VISIBLE_DEVICES=1 python3 split_sub_gqa.py 1 1 &
CUDA_VISIBLE_DEVICES=2 python3 split_sub_gqa.py 2 1 &
CUDA_VISIBLE_DEVICES=3 python3 split_sub_gqa.py 3 1 &
CUDA_VISIBLE_DEVICES=4 python3 split_sub_gqa.py 4 1 &
CUDA_VISIBLE_DEVICES=5 python3 split_sub_gqa.py 5 1 &
CUDA_VISIBLE_DEVICES=6 python3 split_sub_gqa.py 6 1 &
CUDA_VISIBLE_DEVICES=7 python3 split_sub_gqa.py 7 1

sleep 20
CUDA_VISIBLE_DEVICES=0 python3 split_sub_gqa.py 0 2 &
CUDA_VISIBLE_DEVICES=1 python3 split_sub_gqa.py 1 2 &
CUDA_VISIBLE_DEVICES=2 python3 split_sub_gqa.py 2 2 &
CUDA_VISIBLE_DEVICES=3 python3 split_sub_gqa.py 3 2 &
CUDA_VISIBLE_DEVICES=4 python3 split_sub_gqa.py 4 2 &
CUDA_VISIBLE_DEVICES=5 python3 split_sub_gqa.py 5 2 &
CUDA_VISIBLE_DEVICES=6 python3 split_sub_gqa.py 6 2 &
CUDA_VISIBLE_DEVICES=7 python3 split_sub_gqa.py 7 2

sleep 20
CUDA_VISIBLE_DEVICES=0 python3 split_sub_gqa.py 0 3 &
CUDA_VISIBLE_DEVICES=1 python3 split_sub_gqa.py 1 3 &
CUDA_VISIBLE_DEVICES=2 python3 split_sub_gqa.py 2 3 &
CUDA_VISIBLE_DEVICES=3 python3 split_sub_gqa.py 3 3 &
CUDA_VISIBLE_DEVICES=4 python3 split_sub_gqa.py 4 3 &
CUDA_VISIBLE_DEVICES=5 python3 split_sub_gqa.py 5 3 &
CUDA_VISIBLE_DEVICES=6 python3 split_sub_gqa.py 6 3 &
CUDA_VISIBLE_DEVICES=7 python3 split_sub_gqa.py 7 3

sleep 20
CUDA_VISIBLE_DEVICES=0 python3 split_sub_gqa.py 0 4 &
CUDA_VISIBLE_DEVICES=1 python3 split_sub_gqa.py 1 4 &
CUDA_VISIBLE_DEVICES=2 python3 split_sub_gqa.py 2 4 &
CUDA_VISIBLE_DEVICES=3 python3 split_sub_gqa.py 3 4 &
CUDA_VISIBLE_DEVICES=4 python3 split_sub_gqa.py 4 4 &
CUDA_VISIBLE_DEVICES=5 python3 split_sub_gqa.py 5 4 &
CUDA_VISIBLE_DEVICES=6 python3 split_sub_gqa.py 6 4 &
CUDA_VISIBLE_DEVICES=7 python3 split_sub_gqa.py 7 4

sleep 20
CUDA_VISIBLE_DEVICES=0 python3 split_sub_gqa.py 0 5 &
CUDA_VISIBLE_DEVICES=1 python3 split_sub_gqa.py 1 5 &
CUDA_VISIBLE_DEVICES=2 python3 split_sub_gqa.py 2 5 &
CUDA_VISIBLE_DEVICES=3 python3 split_sub_gqa.py 3 5 &
CUDA_VISIBLE_DEVICES=4 python3 split_sub_gqa.py 4 5 &
CUDA_VISIBLE_DEVICES=5 python3 split_sub_gqa.py 5 5 &
CUDA_VISIBLE_DEVICES=6 python3 split_sub_gqa.py 6 5 &
CUDA_VISIBLE_DEVICES=7 python3 split_sub_gqa.py 7 5

sleep 20
CUDA_VISIBLE_DEVICES=0 python3 split_sub_gqa.py 0 6 &
CUDA_VISIBLE_DEVICES=1 python3 split_sub_gqa.py 1 6 &
CUDA_VISIBLE_DEVICES=2 python3 split_sub_gqa.py 2 6 &
CUDA_VISIBLE_DEVICES=3 python3 split_sub_gqa.py 3 6 &
CUDA_VISIBLE_DEVICES=4 python3 split_sub_gqa.py 4 6 &
CUDA_VISIBLE_DEVICES=5 python3 split_sub_gqa.py 5 6 &
CUDA_VISIBLE_DEVICES=6 python3 split_sub_gqa.py 6 6 &
CUDA_VISIBLE_DEVICES=7 python3 split_sub_gqa.py 7 6

sleep 20
CUDA_VISIBLE_DEVICES=0 python3 split_sub_gqa.py 0 7 &
CUDA_VISIBLE_DEVICES=1 python3 split_sub_gqa.py 1 7 &
CUDA_VISIBLE_DEVICES=2 python3 split_sub_gqa.py 2 7 &
CUDA_VISIBLE_DEVICES=3 python3 split_sub_gqa.py 3 7 &
CUDA_VISIBLE_DEVICES=4 python3 split_sub_gqa.py 4 7 &
CUDA_VISIBLE_DEVICES=5 python3 split_sub_gqa.py 5 7 &
CUDA_VISIBLE_DEVICES=6 python3 split_sub_gqa.py 6 7 &
CUDA_VISIBLE_DEVICES=7 python3 split_sub_gqa.py 7 7

sleep 20
CUDA_VISIBLE_DEVICES=0 python3 split_sub_gqa.py 0 8 &
CUDA_VISIBLE_DEVICES=1 python3 split_sub_gqa.py 1 8 &
CUDA_VISIBLE_DEVICES=2 python3 split_sub_gqa.py 2 8 &
CUDA_VISIBLE_DEVICES=3 python3 split_sub_gqa.py 3 8 &
CUDA_VISIBLE_DEVICES=4 python3 split_sub_gqa.py 4 8 &
CUDA_VISIBLE_DEVICES=5 python3 split_sub_gqa.py 5 8 &
CUDA_VISIBLE_DEVICES=6 python3 split_sub_gqa.py 6 8 &
CUDA_VISIBLE_DEVICES=7 python3 split_sub_gqa.py 7 8

sleep 20
CUDA_VISIBLE_DEVICES=0 python3 split_sub_gqa.py 0 9 &
CUDA_VISIBLE_DEVICES=1 python3 split_sub_gqa.py 1 9 &
CUDA_VISIBLE_DEVICES=2 python3 split_sub_gqa.py 2 9 &
CUDA_VISIBLE_DEVICES=3 python3 split_sub_gqa.py 3 9 &
CUDA_VISIBLE_DEVICES=4 python3 split_sub_gqa.py 4 9 &
CUDA_VISIBLE_DEVICES=5 python3 split_sub_gqa.py 5 9 &
CUDA_VISIBLE_DEVICES=6 python3 split_sub_gqa.py 6 9 &
CUDA_VISIBLE_DEVICES=7 python3 split_sub_gqa.py 7 9



# novel testing phase
#
#sleep 20
#CUDA_VISIBLE_DEVICES=0 python3 split_sub_gqa.py 0 10 &
#CUDA_VISIBLE_DEVICES=1 python3 split_sub_gqa.py 1 10 &
#CUDA_VISIBLE_DEVICES=2 python3 split_sub_gqa.py 2 10 &
#CUDA_VISIBLE_DEVICES=3 python3 split_sub_gqa.py 3 10 &
#CUDA_VISIBLE_DEVICES=4 python3 split_sub_gqa.py 4 10 &
#CUDA_VISIBLE_DEVICES=5 python3 split_sub_gqa.py 5 10 &
#CUDA_VISIBLE_DEVICES=6 python3 split_sub_gqa.py 6 10 &
#CUDA_VISIBLE_DEVICES=7 python3 split_sub_gqa.py 7 10
#
#sleep 20
#CUDA_VISIBLE_DEVICES=0 python3 split_sub_gqa.py 0 11 &
#CUDA_VISIBLE_DEVICES=1 python3 split_sub_gqa.py 1 11 &
#CUDA_VISIBLE_DEVICES=2 python3 split_sub_gqa.py 2 11 &
#CUDA_VISIBLE_DEVICES=3 python3 split_sub_gqa.py 3 11 &
#CUDA_VISIBLE_DEVICES=4 python3 split_sub_gqa.py 4 11 &
#CUDA_VISIBLE_DEVICES=5 python3 split_sub_gqa.py 5 11 &
#CUDA_VISIBLE_DEVICES=6 python3 split_sub_gqa.py 6 11 &
#CUDA_VISIBLE_DEVICES=7 python3 split_sub_gqa.py 7 11
#
#sleep 20
#CUDA_VISIBLE_DEVICES=0 python3 split_sub_gqa.py 0 12 &
#CUDA_VISIBLE_DEVICES=1 python3 split_sub_gqa.py 1 12 &
#CUDA_VISIBLE_DEVICES=2 python3 split_sub_gqa.py 2 12 &
#CUDA_VISIBLE_DEVICES=3 python3 split_sub_gqa.py 3 12 &
#CUDA_VISIBLE_DEVICES=4 python3 split_sub_gqa.py 4 12 &
#CUDA_VISIBLE_DEVICES=5 python3 split_sub_gqa.py 5 12 &
#CUDA_VISIBLE_DEVICES=6 python3 split_sub_gqa.py 6 12 &
#CUDA_VISIBLE_DEVICES=7 python3 split_sub_gqa.py 7 12
#
#sleep 20
#CUDA_VISIBLE_DEVICES=0 python3 split_sub_gqa.py 0 13 &
#CUDA_VISIBLE_DEVICES=1 python3 split_sub_gqa.py 1 13 &
#CUDA_VISIBLE_DEVICES=2 python3 split_sub_gqa.py 2 13 &
#CUDA_VISIBLE_DEVICES=3 python3 split_sub_gqa.py 3 13 &
#CUDA_VISIBLE_DEVICES=4 python3 split_sub_gqa.py 4 13 &
#CUDA_VISIBLE_DEVICES=5 python3 split_sub_gqa.py 5 13 &
#CUDA_VISIBLE_DEVICES=6 python3 split_sub_gqa.py 6 13 &
#CUDA_VISIBLE_DEVICES=7 python3 split_sub_gqa.py 7 13
#
#sleep 20
#CUDA_VISIBLE_DEVICES=0 python3 split_sub_gqa.py 0 14 &
#CUDA_VISIBLE_DEVICES=1 python3 split_sub_gqa.py 1 14 &
#CUDA_VISIBLE_DEVICES=2 python3 split_sub_gqa.py 2 14 &
#CUDA_VISIBLE_DEVICES=3 python3 split_sub_gqa.py 3 14 &
#CUDA_VISIBLE_DEVICES=4 python3 split_sub_gqa.py 4 14 &
#CUDA_VISIBLE_DEVICES=5 python3 split_sub_gqa.py 5 14 &
#CUDA_VISIBLE_DEVICES=6 python3 split_sub_gqa.py 6 14 &
#CUDA_VISIBLE_DEVICES=7 python3 split_sub_gqa.py 7 14
#
#sleep 20
#CUDA_VISIBLE_DEVICES=0 python3 split_sub_gqa.py 0 15 &
#CUDA_VISIBLE_DEVICES=1 python3 split_sub_gqa.py 1 15 &
#CUDA_VISIBLE_DEVICES=2 python3 split_sub_gqa.py 2 15 &
#CUDA_VISIBLE_DEVICES=3 python3 split_sub_gqa.py 3 15 &
#CUDA_VISIBLE_DEVICES=4 python3 split_sub_gqa.py 4 15 &
#CUDA_VISIBLE_DEVICES=5 python3 split_sub_gqa.py 5 15 &
#CUDA_VISIBLE_DEVICES=6 python3 split_sub_gqa.py 6 15 &
#CUDA_VISIBLE_DEVICES=7 python3 split_sub_gqa.py 7 15
#
#sleep 20
#CUDA_VISIBLE_DEVICES=0 python3 split_sub_gqa.py 0 16 &
#CUDA_VISIBLE_DEVICES=1 python3 split_sub_gqa.py 1 16 &
#CUDA_VISIBLE_DEVICES=2 python3 split_sub_gqa.py 2 16 &
#CUDA_VISIBLE_DEVICES=3 python3 split_sub_gqa.py 3 16 &
#CUDA_VISIBLE_DEVICES=4 python3 split_sub_gqa.py 4 16 &
#CUDA_VISIBLE_DEVICES=5 python3 split_sub_gqa.py 5 16 &
#CUDA_VISIBLE_DEVICES=6 python3 split_sub_gqa.py 6 16 &
#CUDA_VISIBLE_DEVICES=7 python3 split_sub_gqa.py 7 16
#
#sleep 20
#CUDA_VISIBLE_DEVICES=0 python3 split_sub_gqa.py 0 17 &
#CUDA_VISIBLE_DEVICES=1 python3 split_sub_gqa.py 1 17 &
#CUDA_VISIBLE_DEVICES=2 python3 split_sub_gqa.py 2 17 &
#CUDA_VISIBLE_DEVICES=3 python3 split_sub_gqa.py 3 17 &
#CUDA_VISIBLE_DEVICES=4 python3 split_sub_gqa.py 4 17 &
#CUDA_VISIBLE_DEVICES=5 python3 split_sub_gqa.py 5 17 &
#CUDA_VISIBLE_DEVICES=6 python3 split_sub_gqa.py 6 17 &
#CUDA_VISIBLE_DEVICES=7 python3 split_sub_gqa.py 7 17
#
#sleep 20
#CUDA_VISIBLE_DEVICES=0 python3 split_sub_gqa.py 0 18 &
#CUDA_VISIBLE_DEVICES=1 python3 split_sub_gqa.py 1 18 &
#CUDA_VISIBLE_DEVICES=2 python3 split_sub_gqa.py 2 18 &
#CUDA_VISIBLE_DEVICES=3 python3 split_sub_gqa.py 3 18 &
#CUDA_VISIBLE_DEVICES=4 python3 split_sub_gqa.py 4 18 &
#CUDA_VISIBLE_DEVICES=5 python3 split_sub_gqa.py 5 18 &
#CUDA_VISIBLE_DEVICES=6 python3 split_sub_gqa.py 6 18 &
#CUDA_VISIBLE_DEVICES=7 python3 split_sub_gqa.py 7 18
#
#sleep 20
#CUDA_VISIBLE_DEVICES=0 python3 split_sub_gqa.py 0 19 &
#CUDA_VISIBLE_DEVICES=1 python3 split_sub_gqa.py 1 19 &
#CUDA_VISIBLE_DEVICES=2 python3 split_sub_gqa.py 2 19 &
#CUDA_VISIBLE_DEVICES=3 python3 split_sub_gqa.py 3 19 &
#CUDA_VISIBLE_DEVICES=4 python3 split_sub_gqa.py 4 19 &
#CUDA_VISIBLE_DEVICES=5 python3 split_sub_gqa.py 5 19 &
#CUDA_VISIBLE_DEVICES=6 python3 split_sub_gqa.py 6 19 &
#CUDA_VISIBLE_DEVICES=7 python3 split_sub_gqa.py 7 19

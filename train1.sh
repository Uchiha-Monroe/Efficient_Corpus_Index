#!/bin/bash

torchrun --standalone --nnodes=1 --nproc_per_node=2 train_step1.py \
     &> log.txt
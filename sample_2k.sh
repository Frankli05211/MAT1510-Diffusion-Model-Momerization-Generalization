#!/bin/bash

MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"
NUM_GPUS=1

mpiexec -n $NUM_GPUS python scripts/image_sample.py --model_path model_cps/ema1_rate0.9999_100000.pt $MODEL_FLAGS $DIFFUSION_FLAGS

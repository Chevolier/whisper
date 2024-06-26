#!/bin/bash

pip install --upgrade pip
pip install --upgrade git+https://github.com/huggingface/transformers.git accelerate datasets[audio]

pip install -U openai-whisper
pip install -r requirements.txt

pip install ffmpeg

ngpu=$SM_NUM_GPUS  # number of GPUs to perform distributed training on.

torchrun --nproc_per_node=${ngpu} finetune/train/fine-tune_on_custom_dataset.py \
--model_name whisper-large-v3 \
--language Cantonese \
--sampling_rate 16000 \
--num_proc 1 \
--train_strategy steps \
--learning_rate 5e-6 \
--warmup 10 \
--train_batchsize 8 \
--eval_batchsize 8 \
--num_epochs 10 \
--num_steps 500 \
--resume_from_ckpt None \
--output_dir /opt/ml/checkpoints \
--train_datasets /tmp/data/train \
--eval_datasets /tmp/data/valid
        
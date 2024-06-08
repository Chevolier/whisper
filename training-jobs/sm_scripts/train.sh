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
--num_proc ${ngpu} \
--train_strategy epoch \
--learning_rate 3e-3 \
--warmup 1000 \
--train_batchsize 4 \
--eval_batchsize 4 \
--num_epochs 5 \
--resume_from_ckpt None \
--output_dir /opt/ml/checkpoints \
--train_datasets data/train \
--eval_datasets data/valid
        
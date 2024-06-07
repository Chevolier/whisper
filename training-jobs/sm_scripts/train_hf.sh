#!/bin/bash

pip install --upgrade pip
pip install --upgrade git+https://github.com/huggingface/transformers.git accelerate datasets[audio]

pip install -U openai-whisper
pip install -r requirements.txt

pip install ffmpeg

ngpu=$SM_NUM_GPUS  # number of GPUs to perform distributed training on.

torchrun --nproc_per_node=${ngpu} finetune/train/fine-tune_on_hf_dataset.py \
--model_name /home/ec2-user/SageMaker/efs/Models/whisper-large-v3 \
--language Cantonese \
--sampling_rate 16000 \
--num_proc ${ngpu} \
--train_strategy steps \
--learning_rate 3e-3 \
--warmup 1000 \
--train_batchsize 1 \
--eval_batchsize 1 \
--num_steps 10000 \
--resume_from_ckpt None \
--output_dir /opt/ml/checkpoints \
--train_datasets mozilla-foundation/common_voice_17_0  \
--train_dataset_configs yue \
--train_dataset_splits validation \
--train_dataset_text_columns sentence \
--eval_datasets mozilla-foundation/common_voice_17_0 \
--eval_dataset_configs yue \
--eval_dataset_splits test \
--eval_dataset_text_columns sentence


ngpu=8  # number of GPUs to perform distributed training on.

torchrun --nproc_per_node=${ngpu} finetune/train/fine-tune_on_custom_dataset.py \
--model_name /home/ec2-user/SageMaker/efs/Models/whisper-large-v3 \
--language Cantonese \
--sampling_rate 16000 \
--num_proc 8 \
--train_strategy steps \
--learning_rate 5e-6 \
--warmup 10 \
--train_batchsize 1 \
--eval_batchsize 1 \
--num_epochs 10 \
--num_steps 2000 \
--resume_from_ckpt None \
--output_dir checkpoint/checkpoint-v0 \
--train_datasets data/custom_data_v0/train \
--eval_datasets data/custom_data_v0/valid

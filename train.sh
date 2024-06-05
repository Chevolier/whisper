
ngpu=1  # number of GPUs to perform distributed training on.

torchrun --nproc_per_node=${ngpu} finetune/train/fine-tune_on_custom_dataset.py \
--model_name /home/ec2-user/SageMaker/efs/Models/whisper-large-v3 \
--language Cantonese \
--sampling_rate 16000 \
--num_proc ${ngpu} \
--train_strategy epoch \
--learning_rate 3e-3 \
--warmup 1000 \
--train_batchsize ${train_batch_size} \
--eval_batchsize ${eval_batch_size} \
--num_epochs 2 \
--resume_from_ckpt None \
--output_dir checkpoint \
--train_datasets data/custom_test_data \
--eval_datasets data/custom_test_data
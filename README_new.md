# Environement Configuration

```bash
conda create -n whisper python=3.10
conda activate whisper

pip install --upgrade pip
pip install --upgrade git+https://github.com/huggingface/transformers.git accelerate datasets[audio]

pip install -U openai-whisper
pip install -r requirements.txt

pip install flash-attn --no-build-isolation
conda install ffmpeg
```

# Data Preparation
1. First download 2 open datasets, 

Common Voice Corpus 17.0 (中文（香港）)：https://commonvoice.mozilla.org/zh-HK/datasets
MDCC：https://github.com/HLTCHKUST/cantonese-asr?tab=readme-ov-file

and unzip them in the data/ directory. 

Common Voice directory

cv-corpus-17.0-2024-03-15/
└── zh-HK
    ├── clip_durations.tsv
    ├── clips
    ├── invalidated.tsv
    ├── other.tsv
    ├── reported.tsv
    ├── test.tsv
    ├── train.tsv
    ├── unvalidated_sentences.tsv
    ├── validated_sentences.tsv
    ├── validated.tsv
    └── valid.tsv

MDCC folder
MDCC/
├── audio
├── clip_info_rthk.csv
├── cnt_asr_metadata_full.csv
├── cnt_asr_test_metadata.csv
├── cnt_asr_train_metadata.csv
├── cnt_asr_valid_metadata.csv
├── data_statistic.py
├── length
├── podcast_447_2021.csv
├── README.md
├── test.txt
├── transcription
└── words_length


Put midea data also in the data/ directory in the following format

midea_2173/
├── amrs
└── transcripts.csv


2. Run the commands in data_process.ipynb step by step and generate the custom_data_v7 data, its structure is

custom_data_v7/
├── test
│   ├── audio_paths
│   ├── data-00000-of-00001.arrow
│   ├── dataset_info.json
│   ├── state.json
│   └── text
├── train
│   ├── audio_paths
│   ├── cache-1ec12c05f7b14d0d.arrow
│   ├── cache-b01ce5b489deda03_00000_of_00008.arrow
│   ├── cache-b01ce5b489deda03_00001_of_00008.arrow
│   ├── cache-b01ce5b489deda03_00002_of_00008.arrow
...
│   ├── dataset_info.json
│   ├── state.json
│   └── text
└── valid
    ├── audio_paths
    ├── cache-2081f3f5ea51dc4e_00000_of_00008.arrow
    ├── cache-2081f3f5ea51dc4e_00001_of_00008.arrow
...
    ├── dataset_info.json
    ├── state.json
    └── text


# Training, provide two methods, suggest to use "Training in training-jobs" method since a larger batch size could be used.

## Training in training-jobs

1. First download whisper-large-v3 model from huggingface https://huggingface.co/openai/whisper-large-v3.

2. Go to training-jobs directory and run the command in train.ipynb step by step, note to change the local_model_path = "/home/ec2-user/SageMaker/efs/Models/whisper-large-v3" in Step 2. Upload pretrained models to S3 to the path of downloaded whisper-large-v3 model in your notebook instance. 

Training with 200 steps takes about 4 hours, after that, the checkpoint should be uploaded to the S3 path 

s3://{your-bucket}/checkpoints/whisper_checkpoint_v7

3. Download checkpoint-60 with the following command to the local checkpoint/ directory

```bash
aws s3 sync s3://{your-bucket}/checkpoints/whisper_checkpoint_v7/checkpoint-60/ checkpoint/checkpoint-60 --exclude "*.pth"
```

## Training in notebooks (only when ml.p4d.24xlarge is not available, could try this method)

Suggest to use ml.g5.48xlarge, run the following command

```bash
bash train.sh
```


# Deploy faster whisper to SageMaker endpoint

Go to sagemaker-deploy/faster-whisper/, and run the commands in huggingface.ipynb step by step.











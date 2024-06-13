# Environement Configuration

```bash
conda create -n whisper_py310 python=3.10
conda activate whisper_py310

pip install --upgrade pip
pip install --upgrade git+https://github.com/huggingface/transformers.git accelerate datasets[audio]

pip install -U openai-whisper
pip install -r requirements.txt

pip install flash-attn --no-build-isolation
conda install ffmpeg
```

# Training 

## Training in notebooks


## Training in training-jobs









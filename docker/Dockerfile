# Use the base image
FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:2.3.0-gpu-py311-cu121-ubuntu20.04-sagemaker

# Set the working directory
WORKDIR /app

# Copy requirements.txt to the working directory
COPY requirements.txt .

# Install ffmpeg
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --upgrade git+https://github.com/huggingface/transformers.git accelerate datasets[audio]

RUN pip install -r requirements.txt
    
RUN pip install -U openai-whisper

RUN pip install flash-attn --no-build-isolation

# Copy the rest of your application code to the working directory
COPY . .


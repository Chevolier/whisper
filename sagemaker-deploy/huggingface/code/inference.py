# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0


import os
import io
import sys
import time
import json
import logging

import whisper
import torch
import boto3
import ffmpeg
import torchaudio
import tempfile
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
chunk_length_s = int(os.environ.get('chunk_length_s'))

def model_fn(model_dir):
    # model = pipeline(
    #     "automatic-speech-recognition",
    #     model=model_dir,
    #     chunk_length_s=chunk_length_s,
    #     device=device,
    #     )
    
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_dir, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True) # use_flash_attention_2=True) # attn_implementation="flash_attention_2" only Ampere GPUs support flash attention 
    
    model.to(device)
    
    if hasattr(model.generation_config, "no_timestamps_token_id"):
        return_timestamps = True
    else:
        return_timestamps = False
        
    processor = AutoProcessor.from_pretrained(model_dir)
    
    model = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=chunk_length_s,
        batch_size=16,
        return_timestamps=return_timestamps,
        torch_dtype=torch_dtype,
        device=device,
    )
    
    return model


def transform_fn(model, request_body, request_content_type, response_content_type="application/json"):
     
    logging.info("Check out the request_body type: %s", type(request_body))
    
    start_time = time.time()
    
    print("start reading audio files ...")
    
    file = io.BytesIO(request_body)
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(file.read())

    logging.info("Start to generate the transcription ...")
    result = model(tfile.name)["text"]
    
    logging.info("Upload transcription results back to S3 bucket ...")
    
    # Calculate the elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info("The time for running this program is %s", elapsed_time)
    
    return json.dumps(result), response_content_type
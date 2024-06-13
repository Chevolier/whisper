from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import gradio as gr
import numpy as np
import torch

# model_path = "/home/ec2-user/SageMaker/efs/Models/whisper-large-v3"
model_path = "/home/ec2-user/SageMaker/efs/Projects/whisper/checkpoint/checkpoint-v1-5e6/checkpoint-68"
model_path = "/app/checkpoint/checkpoint-v1-5e6/checkpoint-68"


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
print(f"device: {device}")

processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True, attn_implementation="flash_attention_2") 
model.to(device)
        
transcriber = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

###### asw without streaming

def transcribe(audio):
    sr, y = audio
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    return transcriber({"sampling_rate": sr, "raw": y}, generate_kwargs={"language": "cantonese"})["text"]


demo = gr.Interface(
    transcribe,
    gr.Audio(sources=["microphone", "upload"]),
    "text",
)

demo.launch(share=True)

####### streaming service

# transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")

# def transcribe(stream, new_chunk):
#     sr, y = new_chunk
#     y = y.astype(np.float32)
#     y /= np.max(np.abs(y))

#     if stream is not None:
#         stream = np.concatenate([stream, y])
#     else:
#         stream = y
#     return stream, transcriber({"sampling_rate": sr, "raw": stream})["text"]


# demo = gr.Interface(
#     transcribe,
#     ["state", gr.Audio(sources=["microphone"], streaming=True)],
#     ["state", "text"],
#     live=True,
# )

# demo.launch(share=True)


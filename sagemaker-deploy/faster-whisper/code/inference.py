
from faster_whisper import WhisperModel
import boto3
import json
import os
from decimal import Decimal
def model_fn(model_dir):
    # model_size = "large-v3"
    ct_model_path=".model"
    #model = WhisperModel(model_dir)
    model = WhisperModel(model_dir, device="cuda", compute_type="float16")
    #model = WhisperModel(model_dir, device="cpu", compute_type="int8")
    return model

def transform_fn(model, request_body, request_content_type, response_content_type="application/json"):
    print(f"request_body:{request_body}")
    data = json.loads(request_body)
    print(f"type:{type(data)}")
    s3_client = boto3.client("s3",region_name='us-west-2')  # ap-southeast-1
    s3_bucket = data['s3_bucket']
    print(f"s3_bucket:{type(s3_bucket)}")
    object_key = data['key']
    print(f"object_key:{type(object_key)}")
    audio_file_name = object_key[object_key.rfind('/')+1:]
    contact_id = audio_file_name[0:audio_file_name.find("_")]
    fragment_id = audio_file_name[audio_file_name.find("_")+1:audio_file_name.rfind("_")]
    s3_client.download_file(s3_bucket, object_key, f"/tmp/{audio_file_name}")
    segments, info = model.transcribe(f"/tmp/{audio_file_name}",language="yue",vad_filter=True, vad_parameters=dict(min_silence_duration_ms=100),)
    scripts = []
    speaker =""
    if audio_file_name.find("_agent.wav") > 0:
        speaker = "agent"
    else:
        speaker = "customer"
    for segment in segments:
        scripts.append({"contact_id":contact_id,"time_stamp":int(fragment_id)+segment.start,"words":str(segment.text),"speaker":speaker})
    dynamodb_client = boto3.resource('dynamodb',region_name='us-west-2')
    conversation_table = dynamodb_client.Table("speech2text")
    with conversation_table.batch_writer() as batch:
        for item in scripts:
            #print(item)
            response = batch.put_item(Item={
            "ContactID": item["contact_id"],
            "SaidTimeStamp": Decimal(str(item["time_stamp"])),
            "Speaker": item["speaker"],
            "Words": item["words"]
            })
    #print(f"result:{scripts}")
    os.remove(f"/tmp/{audio_file_name}")
    #return scripts
    return json.dumps(scripts), response_content_type


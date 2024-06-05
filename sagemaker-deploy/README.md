# How to deploy [whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) successfully by SageMaker notebook?


When you try to deploy whisper-large-v3 by SageMaker notebook, you may get error like below:


```
ModelError: An error occurred (ModelError) when calling the InvokeEndpoint operation: Received client error (400) from primary with message "{
  "code": 400,
  "type": "InternalServerException",
  "message": "Wrong index found for \u003c|0.02|\u003e: should be None but found 50366."
 
```

In particular, it seems the AWS Deep Learning Containers only support up to `transformers` version `4.26.0`, which is too low.

You can find details here https://huggingface.co/openai/whisper-large-v3/discussions/58 Therefore, I’m sharing a workaround (custom way) to deploy the model or its derivatives.


1. Create a notebook instance and a notebook file 
2. Execute following python code section by section

```
!pip install -U sagemaker
```

```
!pip install transformers
```

```
!pip list | grep boto3
```

```
import os

# Directory and file paths
dir_path = './model'
inference_file_path = os.path.join(dir_path, 'code/inference.py')
requirements_file_path = os.path.join(dir_path, 'code/requirements.txt')

# Create the directory structure
os.makedirs(os.path.dirname(inference_file_path), exist_ok=True)

# Inference.py content
inference_content = '''
from faster_whisper import WhisperModel
import boto3
import json
import os
from decimal import Decimal
def model_fn(model_dir):
    model_size = "large-v3"
    ct_model_path=".model"
    #model = WhisperModel(model_size)
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    #model = WhisperModel(model_size, device="cpu", compute_type="int8")
    return model

def transform_fn(model, request_body, request_content_type, response_content_type="application/json"):
    print(f"request_body:{request_body}")
    data = json.loads(request_body)
    print(f"type:{type(data)}")
    s3_client = boto3.client("s3",region_name='ap-southeast-1')
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
    dynamodb_client = boto3.resource('dynamodb',region_name='ap-southeast-1')
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

'''

# Write the inference.py file
with open(inference_file_path, 'w') as file:
    file.write(inference_content)

# Requirements.txt content
requirements_content = '''
#transformers==4.36.2
faster_whisper
boto3
ctranslate2==3.24.0
#accelerate==0.26.1
'''
# Write the requirements.txt file
with open(requirements_file_path, 'w') as file:
    file.write(requirements_content)
```

```
import shutil
shutil.make_archive('./model', 'gztar', './model')
```

```
import sagemaker
import boto3
# Get the SageMaker session and default S3 bucket
sagemaker_session = sagemaker.Session()
bucket = "mib-cc-sg-full-open" # Change if you want to store in a different bucket
prefix = 'faster-whisper-large-v3/model'
# Upload the model to S3
s3_path = sagemaker_session.upload_data(
    'model.tar.gz', 
    bucket=bucket,
    key_prefix=prefix
)
print(f"Model uploaded to {s3_path}")
```

```
import boto3
import sagemaker
from sagemaker.huggingface.model import HuggingFaceModel
faster_whisper_big_v3_s3_url = "s3://mib-cc-sg-full-open/faster-whisper-large-v3/model/model.tar.gz"
role = sagemaker.get_execution_role()
print(f"role:{role}")
huggingface_model = HuggingFaceModel(
  model_data=faster_whisper_big_v3_s3_url,
  role=role,
  transformers_version='4.26',
  pytorch_version='1.13',
  py_version='py39',
)
predictor = huggingface_model.deploy(
  initial_instance_count=1,
  instance_type='ml.g4dn.xlarge',
  endpoint_name='fasterwhisperlargev3gpu',
)
```

```
import json
import urllib.parse
import boto3
import os
import logging
ENDPOINT_NAME = 'fasterwhisperlargev3gpu'
AWS_REGION = 'ap-southeast-1'
#dynamodb_client = boto3.resource('dynamodb',region_name='ap-southeast-1')
#s3_client = boto3.client('s3',region_name='ap-southeast-1')
sagemaker_runtime= boto3.client('runtime.sagemaker',region_name='ap-southeast-1')
logger = logging.getLogger()
logger.setLevel("INFO")
#conversation_table = dynamodb_client.Table("speech2text")
s3_bucket = "mib-cc-sg-full-open"
key = urllib.parse.unquote_plus("audio/9ae0a0d7-6e1f-4f22-8e02-9b2c7db59b24_157_agent.wav", encoding='utf-8')
try:
    logger.info("s3_bucket:"+s3_bucket)
    logger.info("key:"+key)
    data = json.dumps({"s3_bucket":s3_bucket,"key":key})
    response = sagemaker_runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,ContentType='application/json',Body=data)
        #logging("type:"+type(response))
        #logging("response:"+str(response))
        #results = json.loads(response['Body'].read().decode())
        #response = s3_client.get_object(Bucket=s3_bucket, Key=key)
    '''
        with conversation_table.batch_writer() as batch:
            for item in results:
                print(item)
                response = batch.put_item(Item={
                "ContactID": item["contact_id"],
                "SaidTimeStamp": item["time_stamp"],
                "Speaker": item["speaker"],
                "Words": item["words"]
                })
        #return response['ContentType']
    '''
    #return "success"
    print(response)
except:
    logger.info("ignore known defects")
```

```
predictor.delete_model()
predictor.delete_endpoint()
```


You may need to slice the audio files by seconds. Here are python to do the job

https://github.com/Mohamedhany99/Audio-Splitter-per-seconds-python-/blob/main/Splitter.py
https://www.geeksforgeeks.org/cut-a-mp3-file-in-python/

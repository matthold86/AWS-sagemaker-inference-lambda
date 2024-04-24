import json
import boto3
import logging
from sagemaker.pytorch import PyTorchPredictor
from sagemaker.deserializers import JSONDeserializer
import time
import sys

# Initialize Logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)

# Initialize the S3 clients
s3_client = boto3.client('s3')
    
def lambda_handler(event, context):
    # Download image from S3
    bucket_name = event['Payload']['bucket_name']
    object_key = event['Payload']['preprocessed_objectkey']
    file_obj = s3_client.get_object(Bucket=bucket_name, Key=object_key)
    payload = file_obj['Body'].read()
    
    # Initialize clients
    ENDPOINT_NAME = "yolov8-serverless-endpoint"
    predictor = PyTorchPredictor(endpoint_name=ENDPOINT_NAME,
                             deserializer=JSONDeserializer())
    infer_start_time = time.time()
    result = predictor.predict(payload)
    infer_end_time = time.time()
    
    logger.info(f"Inference Time = {infer_end_time - infer_start_time:0.4f} seconds")
    logger.info(f"Bounding Box Results: {result}")
                             
    return {
        'statusCode': 200,
        'predictions': json.dumps(result)
    }                             

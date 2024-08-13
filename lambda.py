# Lambda 1: Serializar Imagen

import json
import boto3
import base64

s3 = boto3.client('s3')

def lambda_handler(event, context):
    """A function to serialize target data from S3"""

    # Get S3 key and bucket from Step Function event
    key = event['s3_key']
    bucket = event['s3_bucket']

    # Download the image from S3 to a temporary file
    s3.download_file(bucket, key, "/tmp/image.png")

    # Read the image data and encode it in base64
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')

    # Return the encoded data to the Step Function
    return {
        'statusCode': 200,
        'body': {
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []
        }
    }


# Lambda 2: Image Classification

import json
import sagemaker
import base64
from sagemaker.predictor import Predictor
from sagemaker.serializers import IdentitySerializer

# Name of the deployed endpoint
ENDPOINT = "image-classification-2024-08-12-17-51-12-513"

def lambda_handler(event, context):

    # Decode the image data
    image = base64.b64decode(event['body']['image_data'])

    # Instantiate a SageMaker predictor
    predictor = Predictor(
        endpoint_name=ENDPOINT,
        sagemaker_session=sagemaker.Session()
    )

    # Configure the predictor for PNG images
    predictor.serializer = IdentitySerializer("image/png")

    # Make the prediction
    inferences = predictor.predict(image)

    # Return the inference data to the Step Function
    event["inferences"] = json.loads(inferences.decode('utf-8'))
    return {
        'statusCode': 200,
        'body': event
    }

# Lambda 3: Filter Inferences
import json

# Set a confidence threshold
THRESHOLD = 0.93

def lambda_handler(event, context):

    # Get the event inferences
    inferences = event['body']['inferences']

    # Check if any of the inferences exceed the threshold
    meets_threshold = any([float(x) >= THRESHOLD for x in inferences])

    # If the threshold is met, return the data; otherwise, throw an error
    if meets_threshold:
        return {
            'statusCode': 200,
            'body': event
        }
    else:
        raise Exception("THRESHOLD_CONFIDENCE_NOT_MET")

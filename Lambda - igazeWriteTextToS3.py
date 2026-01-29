import json
import boto3
import urllib.parse

s3 = boto3.client('s3')
polly = boto3.client('polly')

OUTPUT_BUCKET = "s3buckettt"
OUTPUT_PREFIX = "output/"

def lambda_handler(event, context):
    # Get S3 info
    record = event['Records'][0]
    bucket_name = record['s3']['bucket']['name']
    object_key = urllib.parse.unquote_plus(
        record['s3']['object']['key']
    )

    # Read text file from S3
    response = s3.get_object(
        Bucket=bucket_name,
        Key=object_key
    )
    text = response['Body'].read().decode('utf-8')

    # Convert text to speech using Polly
    polly_response = polly.synthesize_speech(
        Text=text,
        OutputFormat='mp3',
        VoiceId='Joanna'
    )

    # Output file name
    file_name = object_key.split('/')[-1].replace('.txt', '.mp3')
    output_key = OUTPUT_PREFIX + file_name

    # Save MP3 to output S3 bucket
    s3.put_object(
        Bucket=OUTPUT_BUCKET,
        Key=output_key,
        Body=polly_response['AudioStream'].read(),
        ContentType='audio/mpeg'
    )

    return {
        'statusCode': 200,
        'body': f'MP3 file saved as {output_key}'
    }

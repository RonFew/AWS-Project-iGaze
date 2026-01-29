import os
import json
import uuid
from datetime import datetime, timezone

import boto3

REGION = os.getenv("REGION", "us-east-2")
BUCKET = os.getenv("BUCKET")
PREFIX = os.getenv("PREFIX", "igaze")
ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "https://igazecycle62.duckdns.org")

s3 = boto3.client("s3", region_name=REGION)
polly = boto3.client("polly", region_name=REGION)


def _ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def _resp(code, obj):
    return {
        "statusCode": code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": ALLOWED_ORIGIN,
            "Access-Control-Allow-Methods": "POST,OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        },
        "body": json.dumps(obj),
    }


def lambda_handler(event, context):
    try:
        if not BUCKET:
            return _resp(500, {"error": "Missing BUCKET environment variable"})

        body = event.get("body") or "{}"
        if isinstance(body, str):
            body = json.loads(body)

        button_id = str(body.get("buttonId", "")).strip()
        text = str(body.get("text", "")).strip()
        voice_id = str(body.get("voiceId", "Joanna")).strip()

        if not button_id or not text:
            return _resp(400, {"error": "buttonId and text are required"})

        if len(text) > 6000:
            return _resp(400, {"error": "text too long (max 6000 characters)"})

        ts = _ts()
        rid = str(uuid.uuid4())

        txt_key = f"{PREFIX}/requests/{ts}_{button_id}_{rid}.txt"
        s3.put_object(
            Bucket=BUCKET,
            Key=txt_key,
            Body=(
                f"timestamp={ts}\n"
                f"buttonId={button_id}\n"
                f"voiceId={voice_id}\n"
                f"text={text}\n"
            ).encode("utf-8"),
            ContentType="text/plain",
        )

        polly_resp = polly.synthesize_speech(
            Text=text, OutputFormat="mp3", VoiceId=voice_id
        )
        audio_bytes = polly_resp["AudioStream"].read()

        mp3_key = f"{PREFIX}/audio/{ts}_{button_id}_{rid}.mp3"
        s3.put_object(
            Bucket=BUCKET,
            Key=mp3_key,
            Body=audio_bytes,
            ContentType="audio/mpeg",
        )

        audio_url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": BUCKET, "Key": mp3_key},
            ExpiresIn=600,
        )

        return _resp(
            200,
            {
                "ok": True,
                "audioUrl": audio_url,
                "audioKey": mp3_key,
                "textKey": txt_key,
            },
        )

    except Exception as e:
        return _resp(500, {"error": str(e)})

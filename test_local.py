import os
import json
import handler  # imports runpod worker

# Force dry-run regardless of shell env
os.environ["SKIP_MODEL_LOAD"] = "1"

payload = {
    "input": {
        "timestamps": False,
        "inputs": [
            {"source": "https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav"}
        ]
    }
}

print(json.dumps(handler.handler(payload), indent=2))

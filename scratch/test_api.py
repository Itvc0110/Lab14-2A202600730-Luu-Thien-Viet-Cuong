import os
import json
import urllib.request
import urllib.error
from dotenv import load_dotenv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=True)

openrouter_key = os.getenv("OPENROUTER_API_KEY", "YOUR_OPENROUTER_API_KEY")

def test_embeddings():
    payload = json.dumps({
        "model": "openai/text-embedding-3-small",
        "input": "test embedding content"
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/embeddings",
        data=payload,
        headers={
            "Authorization": f"Bearer {openrouter_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "Lab14 Eval"
        },
        method="POST"
    )
    print("Testing embeddings on OpenRouter...")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print(f"- Success! Embedding size: {len(data['data'][0]['embedding'])}")
    except Exception as e:
        print(f"- Error: {e}")

test_embeddings()

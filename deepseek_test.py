# deepseek_test.py
import os, requests, json

KEY = os.environ.get("DEEPSEEK_API_KEY") or "MISSING_KEY"
EMBED_URL = os.environ.get("DEEPSEEK_EMBED_URL", "https://api.deepseek.com/v1/embeddings")
MODEL = os.environ.get("DEEPSEEK_EMBED_MODEL", "text-embedding-3-large")

print("EMBED_URL:", EMBED_URL)
print("EMBED_MODEL:", MODEL)
print("KEY present:", KEY != "MISSING_KEY")

payload = {"model": MODEL, "input": ["Hello world"]}
headers = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}

try:
    r = requests.post(EMBED_URL, json=payload, headers=headers, timeout=20)
    print("Status:", r.status_code)
    print("Response:")
    print(r.text[:2000])
except Exception as e:
    print("Exception:", e)

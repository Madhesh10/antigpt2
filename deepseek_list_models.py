# deepseek_list_models.py
import os, requests, json
KEY = os.environ.get("DEEPSEEK_API_KEY") or "MISSING_KEY"
URL = "https://api.deepseek.com/v1/models"
headers = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
try:
    r = requests.get(URL, headers=headers, timeout=20)
    print("Status:", r.status_code)
    print(r.text[:4000])
except Exception as e:
    print("Exception:", e)

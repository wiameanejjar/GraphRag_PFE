import requests

res = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3.1:8b",
        "prompt": "Explique RAG en 1 phrase",
        "stream": False
    }
)

print(res.json()["response"])
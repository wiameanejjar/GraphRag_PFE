import requests
import json

response = requests.post(
    "http://localhost:11434/api/embeddings",
    json={"model": "nomic-embed-text", "prompt": "test"}
)

data = response.json()
embedding = data["embedding"]
print(f"Dimension réelle de nomic-embed-text : {len(embedding)}")
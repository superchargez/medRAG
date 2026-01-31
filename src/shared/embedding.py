# src/shared/embedding.py
import requests
from typing import List
from langchain_core.embeddings import Embeddings

class RemoteEmbeddingFunction(Embeddings):
    """Connector for your custom FastAPI embedding server running on port 8001."""
    def __init__(self, api_url: str):
        self.api_url = api_url

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            response = requests.post(self.api_url, json={"texts": texts})
            if response.status_code != 200:
                raise ValueError(f"Embedding API Error: {response.text}")
            return response.json()["embeddings"]
        except Exception as e:
            print(f"CRITICAL: Failed to connect to Embedding API at {self.api_url}")
            raise e

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]
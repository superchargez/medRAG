# src/shared/embedding.py
import requests
from typing import List, Optional
from langchain_core.embeddings import Embeddings
from src.shared.config import EMBEDDING_PROVIDER

class RemoteEmbeddingFunction(Embeddings):
    """
    Hybrid connector that supports both:
    1. 'llama_local' -> OpenAI compatible format (llama-server)
    2. 'custom'      -> My original Python API format
    """
    def __init__(self, api_url: str, provider: Optional[str] = None):
        self.api_url = api_url
        # If provider not passed explicitly, use the one from config
        self.provider = provider or EMBEDDING_PROVIDER

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Dispatcher: Routing to the correct format based on provider."""
        if self.provider == "llama_local":
            return self._embed_openai_format(texts)
        else:
            return self._embed_custom_format(texts)

    def _embed_openai_format(self, texts: List[str]) -> List[List[float]]:
        """For llama-server /v1/embeddings"""
        payload = {
            "input": texts,
            "model": "default" 
        }
        
        try:
            response = requests.post(self.api_url, json=payload)
            if response.status_code != 200:
                raise ValueError(f"Llama-Server Error: {response.text}")
            
            data = response.json()
            # Sort by index to ensure order is preserved
            sorted_data = sorted(data['data'], key=lambda x: x['index'])
            return [item['embedding'] for item in sorted_data]
            
        except Exception as e:
            print(f"CRITICAL: Failed to connect to Llama-Server at {self.api_url}")
            raise e

    def _embed_custom_format(self, texts: List[str]) -> List[List[float]]:
        """For your original embedding_server.py /embed"""
        try:
            response = requests.post(self.api_url, json={"texts": texts})
            if response.status_code != 200:
                raise ValueError(f"Custom Embedding API Error: {response.text}")
            return response.json()["embeddings"]
        except Exception as e:
            print(f"CRITICAL: Failed to connect to Custom API at {self.api_url}")
            raise e

    def embed_query(self, text: str) -> List[float]:
        """Helper for single query embedding"""
        return self.embed_documents([text])[0]
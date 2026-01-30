# embedding_server.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download
import uvicorn

app = FastAPI(title="MediFlow Embedding Service")

MODEL_ID = "google/embeddinggemma-300m"

# 1. Get the local path dynamically
try:
    # local_files_only=True forces it to look at your disk and NOT the internet.
    # It returns the exact path you saw in your terminal.
    model_path = snapshot_download(repo_id=MODEL_ID, local_files_only=True)
    print(f"Using locally cached model at: {model_path}")
except Exception as e:
    # If for some reason it's NOT there, this is your fallback
    print("Model not found locally, attempting to download...")
    model_path = snapshot_download(repo_id=MODEL_ID)

# 2. Load the model from that dynamic path
print(f"Loading model into memory...")
model = SentenceTransformer(model_path) 
print("Model loaded and ready.")

# 2. Define Request Schema
class EmbeddingRequest(BaseModel):
    texts: List[str]

# 3. Define the Endpoint
@app.post("/embed")
async def embed_texts(request: EmbeddingRequest):
    """
    Accepts a list of strings, returns a list of vector lists.
    """
    try:
        # Generate embeddings
        # convert_to_tensor=False returns numpy arrays, we convert to list for JSON
        embeddings = model.encode(request.texts).tolist()
        return {"embeddings": embeddings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
# src/shared/helpers.py

import hashlib
import json

def sha256_encoder(text: str) -> str:
    """Uses SHA-256 for LangChain indexing to avoid SHA-1 warnings."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def clean_and_parse_json(text: str):
    """
    Cerebras often wraps JSON in markdown blocks (```json ... ```).
    This function strips them and parses the object.
    """
    text = text.strip()
    
    # Remove markdown code blocks
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    
    if text.endswith("```"):
        text = text[:-3]
        
    text = text.strip()
    return json.loads(text)
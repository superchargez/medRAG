# src/shared/llm_factory.py
import os
from langchain_cerebras import ChatCerebras
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from src.shared.config import (
    CEREBRAS_API_KEY, 
    GROQ_API_KEY, 
    GOOGLE_API_KEY, 
    LLM_MODEL_ID,
    LLM_PROVIDER
)

def get_llm_chain(provider=None, temperature=0.1):
    """
    Factory to return a LangChain Chat Model based on provider.
    If provider is None, uses LLM_PROVIDER from config.
    """
    if provider is None:
        provider = LLM_PROVIDER
    
    if provider == "cerebras":
        if not CEREBRAS_API_KEY:
            raise ValueError("Cerebras API Key is missing.")
        
        return ChatCerebras(
            model=LLM_MODEL_ID,
            api_key=CEREBRAS_API_KEY,
            temperature=temperature,
            max_retries=2,
        )

    elif provider == "groq":
        if not GROQ_API_KEY:
            raise ValueError("Groq API Key is missing.")
        
        return ChatGroq(
            model=LLM_MODEL_ID,
            api_key=GROQ_API_KEY,
            temperature=temperature
        )
    
    elif provider == "gemini":
        if not GOOGLE_API_KEY:
            raise ValueError("Google API Key is missing.")
        
        return ChatGoogleGenerativeAI(
            model=LLM_MODEL_ID,
            api_key=GOOGLE_API_KEY,
            temperature=temperature
        )
    
    else:
        raise ValueError(f"Provider {provider} not supported yet. Available: cerebras, groq, gemini")
# src/shared/llm_factory.py
import os
from langchain_cerebras import ChatCerebras
# Uncomment these when you are ready to use them:
# from langchain_groq import ChatGroq
# from langchain_google_genai import ChatGoogleGenerativeAI

from src.shared.config import CEREBRAS_API_KEY, LLM_MODEL_ID

def get_llm_chain(provider="cerebras", temperature=0.1):
    """
    Factory to return a LangChain Chat Model based on provider.
    """
    if provider == "cerebras":
        if not CEREBRAS_API_KEY:
            raise ValueError("Cerebras API Key is missing.")
        
        return ChatCerebras(
            model=LLM_MODEL_ID,
            api_key=CEREBRAS_API_KEY,
            temperature=temperature,
            max_retries=2,
        )

    # --- FUTURE INTEGRATIONS (Placeholders) ---
    # elif provider == "groq":
    #     return ChatGroq(
    #         model="openai/gpt-oss-120b",
    #         temperature=temperature
    #     )
    # elif provider == "gemini":
    #     return ChatGoogleGenerativeAI(
    #         model="gemini-2.5-flash",
    #         temperature=temperature
    #     )
    
    else:
        raise ValueError(f"Provider {provider} not supported yet.")
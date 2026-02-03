# src/shared/llm.py
import requests
import json
from cerebras.cloud.sdk import Cerebras
from groq import Groq # ADD THIS
from langchain_google_genai import ChatGoogleGenerativeAI # ADD THIS

# Update config import to include the new keys
from src.shared.config import (
    CEREBRAS_API_KEY, 
    GROQ_API_KEY, 
    GOOGLE_API_KEY, 
    LLM_PROVIDER, 
    LOCAL_LLM_URL,
    LLM_MODEL_ID
)

# --- Mock Classes (Keep existing) ---
class MockMessage:
    def __init__(self, content):
        self.content = content

class MockChoice:
    def __init__(self, content):
        self.message = MockMessage(content)

class MockResponse:
    def __init__(self, content):
        self.choices = [MockChoice(content)]

# --- 1. Local Llama Client (Keep existing) ---
class LocalLlamaClient:
    # ... (Keep your existing LocalLlamaClient code exactly as is) ...
    def __init__(self, base_url):
        self.base_url = base_url
        self.chat = self._Chat(self)

    class _Chat:
        def __init__(self, parent):
            self.completions = self._Completions(parent)

        class _Completions:
            def __init__(self, parent):
                self.parent = parent

            def create(self, messages, model=None, temperature=0.1, max_completion_tokens=256, extra_body=None, **kwargs):
                prompt_text = ""
                for msg in messages:
                    prompt_text += f"{msg['role'].upper()}: {msg['content']}\n"
                prompt_text += "\nASSISTANT:"

                payload = {
                    "prompt": prompt_text,
                    "n_predict": max_completion_tokens,
                    "temperature": temperature,
                    "stop": ["User:", "System:"],
                    "cache_prompt": True 
                }

                if extra_body and "grammar" in extra_body:
                    print("   [LocalLLM] 🔒 Enforcing JSON Grammar")
                    payload["grammar"] = extra_body["grammar"]

                try:
                    response = requests.post(
                        self.parent.base_url, 
                        headers={"Content-Type": "application/json"},
                        json=payload
                    )
                    
                    if response.status_code != 200:
                        print(f"Error: {response.text}")
                        return MockResponse("{}") 

                    data = response.json()
                    answer = data.get("content", "").strip()
                    return MockResponse(answer)

                except Exception as e:
                    print(f"Local LLM Error: {e}")
                    return MockResponse("{}")


# --- 2. Gemini Client Adapter (ADD THIS) ---
class GeminiClient:
    """
    Adapter to make Google Gemini look like an OpenAI-style client
    so it works with the existing api.py code structure.
    """
    def __init__(self, api_key):
        self.llm = ChatGoogleGenerativeAI(model=LLM_MODEL_ID, api_key=api_key)
        self.chat = self._Chat(self)

    class _Chat:
        def __init__(self, parent):
            self.completions = self._Completions(parent)

        class _Completions:
            def __init__(self, parent):
                self.parent = parent

            def create(self, messages, model=None, temperature=0.1, max_completion_tokens=256, **kwargs):
                # 1. Format messages for LangChain
                # Simple conversion: System + User -> Single prompt or list of HumanMessages
                # For simplicity with your existing prompts, we often concatenate or pass list
                
                langchain_messages = []
                for m in messages:
                    role = m["role"]
                    content = m["content"]
                    if role == "system":
                        langchain_messages.append(("system", content))
                    elif role == "user":
                        langchain_messages.append(("user", content))
                    elif role == "assistant":
                        langchain_messages.append(("assistant", content))

                # 2. Invoke
                try:
                    response = self.parent.llm.invoke(langchain_messages)
                    # 3. Return in OpenAI-style format
                    return MockResponse(response.content)
                except Exception as e:
                    print(f"Gemini Error: {e}")
                    return MockResponse("{}")


# --- 3. Factory Function (UPDATE THIS) ---
def get_cerebras_client():
    if LLM_PROVIDER == "local":
        print(f"--- 🟢 USING LOCAL LLM (Llama.cpp) at {LOCAL_LLM_URL} ---")
        return LocalLlamaClient(base_url=LOCAL_LLM_URL)
    
    elif LLM_PROVIDER == "groq":
        if not GROQ_API_KEY:
            raise ValueError("Groq API Key missing.")
        print(f"--- 🟡 USING GROQ ({LLM_MODEL_ID}) ---")
        return Groq(api_key=GROQ_API_KEY)

    elif LLM_PROVIDER == "gemini":
        if not GOOGLE_API_KEY:
            raise ValueError("Google API Key missing.")
        print(f"--- 🔴 USING GEMINI ({LLM_MODEL_ID}) ---")
        return GeminiClient(api_key=GOOGLE_API_KEY)

    # Default to Cerebras
    elif LLM_PROVIDER == "cerebras":
        if not CEREBRAS_API_KEY:
            raise ValueError("Cerebras API Key missing.")
        print(f"--- 🔵 USING CEREBRAS ({LLM_MODEL_ID}) ---")
        return Cerebras(api_key=CEREBRAS_API_KEY)
    
    else:
        raise ValueError(f"Provider '{LLM_PROVIDER}' not supported.")
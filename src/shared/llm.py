# src/core/llm.py
import requests
import json
from cerebras.cloud.sdk import Cerebras
from src.shared.config import CEREBRAS_API_KEY, LLM_PROVIDER, LOCAL_LLM_URL

# --- 1. Define Mock Classes to Mimic OpenAI/Cerebras Response Structure ---
class MockMessage:
    def __init__(self, content):
        self.content = content

class MockChoice:
    def __init__(self, content):
        self.message = MockMessage(content)

class MockResponse:
    def __init__(self, content):
        self.choices = [MockChoice(content)]

# --- 2. Define the Local Client Adapter ---
class LocalLlamaClient:
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
                # 1. Format Prompt
                prompt_text = ""
                for msg in messages:
                    prompt_text += f"{msg['role'].upper()}: {msg['content']}\n"
                prompt_text += "\nASSISTANT:"

                # 2. Build Payload
                payload = {
                    "prompt": prompt_text,
                    "n_predict": max_completion_tokens,
                    "temperature": temperature,
                    "stop": ["User:", "System:"],
                    "cache_prompt": True 
                }

                # 3. INJECT GRAMMAR IF PROVIDED
                # The router will pass "grammar" inside extra_body
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
                        return MockResponse("{}") # Return empty JSON string on failure

                    data = response.json()
                    answer = data.get("content", "").strip()
                    return MockResponse(answer)

                except Exception as e:
                    print(f"Local LLM Error: {e}")
                    return MockResponse("{}")


# --- 3. Factory Function to Switch Providers ---
def get_cerebras_client():
    """
    Returns either the real Cerebras client or the Local Adapter 
    based on config.LLM_PROVIDER.
    """
    if LLM_PROVIDER == "local":
        print(f"--- 🟢 USING LOCAL LLM (Llama.cpp) at {LOCAL_LLM_URL} ---")
        return LocalLlamaClient(base_url=LOCAL_LLM_URL)
    
    # Default to Cerebras
    if not CEREBRAS_API_KEY:
        raise ValueError("Cerebras API Key missing, and provider set to 'cerebras'")
    
    print("--- 🔵 USING CEREBRAS CLOUD API ---")
    return Cerebras(api_key=CEREBRAS_API_KEY)
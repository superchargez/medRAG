# src/v1_core/router.py
from src.shared.llm import get_cerebras_client
from src.shared.config import LLM_MODEL_ID, LLM_PROVIDER
from src.prompts.templates import (
    ROUTER_SYSTEM_PROMPT, 
    ROUTER_PROMPT_LOCAL, 
    ROUTER_GRAMMAR
)
from src.shared.helpers import clean_and_parse_json
from pydantic import BaseModel, Field
from typing import Literal, Optional, List
from src.prompts.templates import ROUTER_SYSTEM_PROMPT

class RouterDecision(BaseModel):
    route: Literal["patient_info", "medical_wiki", "clinical_safety", "general"]
    # RENAME or REDEFINE semantics: this field now holds Name, ID, Phone, or CNIC
    patient_name: Optional[str] = Field(None, description="The specific name, ID, phone number, or CNIC mentioned in the query.")
    medical_terms: List[str] = []
    reasoning: str = ""

def get_routing_decision(user_query: str) -> RouterDecision:
    client = get_cerebras_client()
    
    # --- LOGIC SWITCH ---
    if LLM_PROVIDER == "local":
        # Local: Use simplified prompt + Grammar
        system_content = ROUTER_PROMPT_LOCAL
        extra_body = {"grammar": ROUTER_GRAMMAR}
        temperature = 0.1
    else:
        # Cerebras: Use explicit JSON instruction prompt
        system_content = ROUTER_SYSTEM_PROMPT
        extra_body = None
        temperature = 0.1

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT + "\nNOTE: If the user provides a number (ID, CNIC, Phone), extract it as 'patient_name'."},
                {"role": "user", "content": f"User Query: {user_query}"}
            ],
            model=LLM_MODEL_ID,
            temperature=temperature,
            max_completion_tokens=512,
            extra_body=extra_body # This is key for the local client
        )
        
        raw_text = response.choices[0].message.content
        
        # Debugging: See what the small model actually output
        print(f"   [Router] Raw Output: {raw_text}")

        data = clean_and_parse_json(raw_text)
        return RouterDecision(**data)

    except Exception as e:
        print(f"Router Error: {e}")
        # Fallback: If routing fails but query is short, assume the whole query might be the identifier
        return RouterDecision(
            route="general", 
            patient_name=user_query if len(user_query) < 20 else None,
            reasoning=f"Error: {e}"
        )
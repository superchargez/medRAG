# src/v3_core/agent.py
from langchain.agents import create_agent
from src.shared.llm_factory import get_llm_chain
from src.shared.config import LLM_PROVIDER 
from src.v3_core.tools import (
    IdentityTool, 
    MedicalSQLAgentTool,
    MedicalWikiTool
)

# Define the toolset
tools = [
    IdentityTool(), 
    MedicalSQLAgentTool(),
    MedicalWikiTool()
]

SYSTEM_PROMPT = """You are MediFlow Agent V3 (Turso-Powered).
Current Date: {date}

INSTRUCTIONS:
1. **Complex Queries:** For ANY question involving counting, filtering by age, dates, conditions, or finding specific patient lists, use the `medical_database_analyzer`. Pass the FULL user question to it.
2. **Identity:** Use `lookup_patient_id` if the user asks "Who is PT-XXX?" or "Find ID for...".
3. **General Knowledge:** Use `check_drug_info` for medical definitions or drug info.

Do not try to answer database questions from your own memory. Always query the database.
"""

def get_agent_graph():
    # Use the provider from config instead of hardcoding "cerebras"
    llm = get_llm_chain(provider=LLM_PROVIDER) 
    graph = create_agent(llm, tools=tools)
    return graph
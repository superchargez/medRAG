# src/v2_core/agent.py
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime

# Imports
from src.shared.llm_factory import get_llm_chain
from src.v2_core.tools import (
    IdentityTool, 
    PatientHistoryTool, 
    PatientAnalyticsTool,
    MedicalWikiTool
)

tools = [
    IdentityTool(), 
    PatientHistoryTool(), 
    PatientAnalyticsTool(),
    MedicalWikiTool()
]

SYSTEM_PROMPT = """You are MediFlow Agent V2 (SQL-Powered).
Current Date: {date}

RULES:
1. **Dates:** ALWAYS convert dates to 'YYYY-MM-DD' format before calling tools.
2. **Analytics:** Use 'patient_analytics_tool' for counting or listing patients.
3. **Identity:** Use 'lookup_patient_id' to find PT-XXX IDs.
4. **History:** Use 'get_patient_medical_history'.
5. **Knowledge:** Use 'check_drug_info' for general medical questions.

Be precise.
"""

def get_agent_graph():
    llm = get_llm_chain(provider="cerebras")
    graph = create_react_agent(llm, tools=tools)
    return graph
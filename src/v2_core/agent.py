# src/v2_core/agent.py

from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime

# Imports
from src.shared.llm_factory import get_llm_chain
from src.v2_core.tools import (
    IdentityTool, 
    PatientHistoryTool, 
    MedicalWikiTool, 
    PatientAnalyticsTool
)

# 1. Initialize Tools
tools = [
    IdentityTool(), 
    PatientHistoryTool(), 
    MedicalWikiTool(), 
    PatientAnalyticsTool()
]

# 2. System Prompt
# We inject the date dynamically in the API, but here is the base template.
SYSTEM_PROMPT = """You are MediFlow Agent V2.
Current Date: {date}

STRATEGY:
1. **Specific Patient:** If user asks about a person, use 'lookup_patient_id' first.
2. **Analytics/Counts:** If user asks "How many...", "List...", "Who has...", use 'patient_analytics_tool'.
   - This tool returns JSON. Read the JSON to answer the user.
   - Do NOT call 'lookup_patient_id' for every person in the list.
3. **Medical Info:** Use 'check_drug_info' for general questions.

Always be concise.
"""

def get_agent_graph():
    """
    Returns the compiled LangGraph runnable.
    """
    # Get LLM (Cerebras)
    llm = get_llm_chain(provider="cerebras")
    
    # Create the Graph
    # 'create_react_agent' automatically binds tools and handles the ReAct loop
    graph = create_agent(llm, tools=tools)
    
    return graph
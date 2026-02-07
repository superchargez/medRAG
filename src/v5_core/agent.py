# src/v5_core/agent.py
from langchain.agents import create_agent
from src.shared.llm_factory import get_llm_chain
from src.shared.config import LLM_PROVIDER
from src.v5_core.tools import (
    CohortFinderTool,
    PatientConditionTool,
    DrugSafetyTool,
    IdentityTool,
    SystemStatsTool
)

tools = [
    CohortFinderTool(),
    PatientConditionTool(),
    DrugSafetyTool(),
    IdentityTool(),
    SystemStatsTool()
]

SYSTEM_PROMPT = """You are MediFlow Agent V5.
Current Date: {date}

You have 5 specialized tools. Use them logically.

--- TOOLS ---
1. `find_patient_cohort`: Filter patients by Age, Gender, Visit Date.
2. `analyze_patient_condition`: Check specific patients' notes for conditions (Vector Search).
3. `check_drug_safety`: Check Wiki for drug info.
4. `lookup_patient_identity`: Find a person by name.
5. `get_system_stats`: Use this for "Meta" questions like "How many files?", "How many chunks?", "Total patients?".

--- RULES ---
1. **Drug Safety:** Always check `check_drug_safety` BEFORE checking patient notes if the question is about medication safety.
2. **Meta Questions:** If asked about "files", "chunks", or "database stats", use `get_system_stats`.
3. **Calculations:** You must calculate percentages yourself based on the tool outputs.
"""

def get_agent_graph():
    llm = get_llm_chain(provider=LLM_PROVIDER, temperature=0) 
    graph = create_agent(llm, tools=tools, system_prompt=SYSTEM_PROMPT)
    return graph
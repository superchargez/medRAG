# src/v2_core/state.py
from typing import Annotated, List, TypedDict, Union
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage

class AgentState(TypedDict):
    # 'add_messages' handles appending new messages to history automatically
    messages: Annotated[List[AnyMessage], add_messages]
    patient_id: Union[str, None]
    patient_name: Union[str, None]
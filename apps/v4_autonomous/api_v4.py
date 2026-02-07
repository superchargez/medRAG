# apps/v4_turso/api_v4.py
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from datetime import datetime

# Import the V4 Graph
from src.v4_core.agent_turso import get_agent_graph, SYSTEM_PROMPT

app = FastAPI(title="MediFlow Agentic API (V4 - Semantic SQL)")

# Initialize the graph once on startup
agent_graph = get_agent_graph()

class AgentRequest(BaseModel):
    query: str

@app.post("/agent/chat")
async def agent_chat(request: AgentRequest):
    print(f"\n[V4 AGENT] Query: {request.query}")
    
    # 1. Inject Date into System Prompt
    today_str = datetime.now().strftime("%Y-%m-%d")
    try:
        formatted_system_prompt = SYSTEM_PROMPT.format(date=today_str)
    except KeyError:
        # Fallback if {date} is missing in prompt template
        formatted_system_prompt = SYSTEM_PROMPT
    
    # 2. Prepare Inputs
    # We override the system prompt dynamically here
    inputs = {
        "messages": [
            {"role": "system", "content": formatted_system_prompt},
            {"role": "user", "content": request.query}
        ]
    }

    try:
        # 3. Invoke Agent
        result = await agent_graph.ainvoke(inputs)
        
        # 4. Extract Answer
        last_message = result["messages"][-1]
        return {
            "status": "success",
            "answer": last_message.content
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Agent Error: {e}")

        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    # Running on port 8004
    uvicorn.run(app, host="0.0.0.0", port=8004)
# apps/v3_turso/api_v3.py
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from datetime import datetime

# Import the V3 Graph
from src.v3_core.agent import get_agent_graph, SYSTEM_PROMPT

app = FastAPI(title="MediFlow Agentic API (V3 - Turso)")

agent_graph = get_agent_graph()

class AgentRequest(BaseModel):
    query: str

@app.post("/agent/chat")
async def agent_chat(request: AgentRequest):
    print(f"\n[V3 TURSO AGENT] Query: {request.query}")
    
    today_str = datetime.now().strftime("%Y-%m-%d")
    formatted_system_prompt = SYSTEM_PROMPT.format(date=today_str)
    
    inputs = {
        "messages": [
            {"role": "system", "content": formatted_system_prompt},
            {"role": "user", "content": request.query}
        ]
    }

    try:
        result = await agent_graph.ainvoke(inputs)
        last_message = result["messages"][-1]
        return {
            "status": "success",
            "answer": last_message.content
        }
    except Exception as e:
        print(f"Agent Error: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8004)
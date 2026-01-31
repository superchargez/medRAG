# apps/v2_agentic/api_v2.py
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from datetime import datetime

# Import the Graph
from src.v2_core.agent import get_agent_graph, SYSTEM_PROMPT

app = FastAPI(title="MediFlow Agentic API (V2)")

# Initialize Graph
agent_graph = get_agent_graph()

class AgentRequest(BaseModel):
    query: str

@app.post("/agent/chat")
async def agent_chat(request: AgentRequest):
    print(f"\n[V2 AGENT] Query: {request.query}")
    
    # 1. Prepare Input
    # We prepend the System Prompt with the current date
    today_str = datetime.now().strftime("%Y-%m-%d")
    formatted_system_prompt = SYSTEM_PROMPT.format(date=today_str)
    
    inputs = {
        "messages": [
            {"role": "system", "content": formatted_system_prompt},
            {"role": "user", "content": request.query}
        ]
    }

    try:
        # 2. Invoke Graph
        # The graph returns the final state. The last message is the answer.
        result = await agent_graph.ainvoke(inputs)
        
        last_message = result["messages"][-1]
        answer = last_message.content
        
        return {
            "status": "success",
            "answer": answer
        }
        
    except Exception as e:
        print(f"Agent Error: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
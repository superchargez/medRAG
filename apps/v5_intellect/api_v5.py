# apps/v5_intellect/api_v5.py
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from datetime import datetime
from langchain_core.callbacks import BaseCallbackHandler

# Import the V5 Graph
from src.v5_core.agent import get_agent_graph, SYSTEM_PROMPT

app = FastAPI(title="MediFlow Agentic API (V5 - Turso)")

agent_graph = get_agent_graph()

class AgentRequest(BaseModel):
    query: str

# --- CUSTOM CALLBACK FOR LOGGING ---
class ConsoleCallbackHandler(BaseCallbackHandler):
    """Prints agent thoughts and tool inputs to the console."""
    def on_tool_start(self, serialized, input_str, **kwargs):
        print(f"\n\033[95m🤖 AGENT THOUGHT: Calling tool '{serialized['name']}' with input:\033[0m")
        print(f"\033[95m{input_str}\033[0m")

@app.post("/agent/chat")
async def agent_chat(request: AgentRequest):
    print(f"\n\n{'='*50}")
    print(f"[V5 AGENT] New Query: {request.query}")
    print(f"{'='*50}")
    
    today_str = datetime.now().strftime("%Y-%m-%d")
    formatted_system_prompt = SYSTEM_PROMPT.format(date=today_str)
    
    inputs = {
        "messages": [
            {"role": "system", "content": formatted_system_prompt},
            {"role": "user", "content": request.query}
        ]
    }

    try:
        # Pass the callback handler to see the logs!
        result = await agent_graph.ainvoke(
            inputs, 
            config={"callbacks": [ConsoleCallbackHandler()]}
        )
        
        last_message = result["messages"][-1]
        return {
            "status": "success",
            "answer": last_message.content
        }
    except Exception as e:
        print(f"Agent Error: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005)
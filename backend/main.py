# backend/main.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from agent import Agent
import os
import uvicorn

app = FastAPI()
agent = Agent()

# Allow requests from your React frontend (update origin if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/list-llamacpp-models")
def list_llamacpp_models(dir: str = Query(...)):
    try:
        files = os.listdir(dir)
        # Filter for .gguf files (or .bin, etc. as needed)
        models = [f for f in files if f.lower().endswith('.gguf')]
        return {"models": models}
    except Exception as e:
        return {"models": [], "error": str(e)}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Wait for a request
            request_data = await websocket.receive_json()

            # Extract data and configure agent
            user_input = request_data.get("userInput")
            settings = request_data.get("settings")
            agent.configure(settings)

            # Prepare initial graph state with websocket for node communication
            initial_state = {
                "messages": [HumanMessage(content=user_input)],
                "user_input": user_input,
                "websocket": websocket,
                "settings": settings
            }

            # Run graph asynchronously
            async for _ in agent.graph.astream(initial_state):
                pass
    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        print(f"WEBSOCKET ERROR: {e}")
        await websocket.send_json({"type": "error", "message": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
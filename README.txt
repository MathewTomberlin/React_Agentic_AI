# React and FastAPI Agentic AI
This project implements an Agentic AI with a React frontend and FastAPI websocket backend.

# Structure
- backend
- - server.ts: MCP Server
- - main.py: FastAPI server
- - agent.py: LangGraph agent
- frontend
- - src: React frontend components

# Design
The LangGraph agent runs and sends updates to and receives commands from the React frontend via the FastAPI server.
The LangGraph agent passes the WebSocket along through its nodes to maintain the connection.

# Notes
The Danbooru Tag LoRA option is disabled because this project doesn't include the LoRA
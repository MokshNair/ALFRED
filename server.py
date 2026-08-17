import uuid
from contextlib import asynccontextmanager
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel

from agent import build_agent_executor, build_llm, is_ollama_available, log_sft_turn, maybe_save_output

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    with SqliteSaver.from_conn_string("ALFRED_memory.db") as checkpointer:
        agents = {"local": build_agent_executor(build_llm(True), checkpointer)}
        try:
            agents["openrouter"] = build_agent_executor(build_llm(False), checkpointer)
        except ValueError as e:
            print(f"[SERVER] OpenRouter unavailable: {e}")

        app.state.agents = agents
        app.state.sessions = {}
        yield


app = FastAPI(title="ALFRED", lifespan=lifespan)


class NewSessionRequest(BaseModel):
    model: Literal["local", "openrouter"]


class ChatRequest(BaseModel):
    thread_id: str
    message: str


def _get_session_model(request: Request, thread_id: str) -> str:
    sessions = request.app.state.sessions
    if thread_id not in sessions:
        raise HTTPException(status_code=404, detail="Unknown session. Start a new one via /session/new.")
    return sessions[thread_id]


def _serialize_history(messages) -> list[dict]:
    history = []
    for m in messages:
        if isinstance(m, HumanMessage):
            history.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage) and m.content:
            history.append({"role": "assistant", "content": m.content})
    return history


@app.post("/session/new")
def new_session(req: NewSessionRequest, request: Request):
    if req.model not in request.app.state.agents:
        raise HTTPException(
            status_code=503,
            detail=f"Model '{req.model}' is not configured on this server (missing API key?).",
        )

    if req.model == "local" and not is_ollama_available():
        raise HTTPException(
            status_code=503,
            detail="Local model unavailable — start Ollama with 'ollama serve' and try again.",
        )

    thread_id = uuid.uuid4().hex
    request.app.state.sessions[thread_id] = req.model
    return {"thread_id": thread_id, "model": req.model}


@app.post("/chat")
def chat(req: ChatRequest, request: Request):
    model = _get_session_model(request, req.thread_id)
    agent_executor = request.app.state.agents[model]
    config = {"configurable": {"thread_id": req.thread_id}}

    try:
        final_state = agent_executor.invoke(
            {"messages": [("user", req.message)]},
            config=config,
        )
        response_text = final_state["messages"][-1].content
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"ALFRED encountered an error: {e}")

    log_sft_turn(req.message, response_text)
    saved = maybe_save_output(req.message, response_text)

    return {"response": response_text, "saved": saved}


@app.get("/history/{thread_id}")
def history(thread_id: str, request: Request):
    model = _get_session_model(request, thread_id)
    agent_executor = request.app.state.agents[model]
    config = {"configurable": {"thread_id": thread_id}}

    state = agent_executor.get_state(config)
    messages = state.values.get("messages", []) if state and state.values else []

    return {"model": model, "messages": _serialize_history(messages)}


app.mount("/", StaticFiles(directory="static", html=True), name="static")

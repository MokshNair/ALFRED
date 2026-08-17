import datetime
import json
import os

import requests
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from tools import knowledge_tool, search_tool, wiki_tool

OLLAMA_BASE_URL = "http://localhost:11434"
LOCAL_MODEL = "qwen2.5:14b"
OPENROUTER_MODEL = "poolside/laguna-s-2.1:free"


def load_persona() -> str:
    try:
        with open("PERSONA.md", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""


PERSONA = load_persona()

OPERATIONAL_RULES = """
Operational Rules:
1. For greetings, casual conversation, introductions, or exit requests, respond directly, honestly, and concisely as a persona. DO NOT use the research template for these interactions.
2. If a query requires factual tracking or data lookups, natively trigger your tools.
3. For questions about my personal notes, study material, AI/ML course, or saved documents, ALWAYS use the search_knowledge_base tool first.
4. ONLY after executing tools to gather research, you MUST format your final synthesis exactly using this structure:

### Complete Research Findings
[Your highly detailed, comprehensive list and explanation of the gathered facts here.]

### Tools Used
[List the tools used]

### Sources
[List the specific URLs]
"""

SYSTEM_PROMPT = f"{PERSONA}\n\n{OPERATIONAL_RULES}".strip()

TOOLS = [search_tool, wiki_tool, knowledge_tool]

SFT_LOG_FILE = "sft_training_data.jsonl"
SAVE_OUTPUT_FILE = "research_output.txt"
SAVE_KEYWORDS = ("save", "write to file", "store", "export")


def is_ollama_available() -> bool:
    try:
        requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        return True
    except requests.exceptions.RequestException:
        return False


def build_llm(use_local: bool) -> ChatOpenAI:
    if use_local:
        return ChatOpenAI(
            base_url=f"{OLLAMA_BASE_URL}/v1",
            api_key="ollama",
            model=LOCAL_MODEL,
            temperature=0,
        )

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_key:
        raise ValueError("CRITICAL: OPENROUTER_API_KEY is missing from your .env file!")

    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_key,
        model=OPENROUTER_MODEL,
        temperature=0,
    )


def build_agent_executor(llm: ChatOpenAI, checkpointer):
    return create_agent(
        model=llm,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )


def log_sft_turn(query: str, response: str) -> None:
    try:
        sft_turn = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT.strip()},
                {"role": "user", "content": query.strip()},
                {"role": "assistant", "content": response.strip()},
            ]
        }
        with open(SFT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(sft_turn) + "\n")
    except Exception as log_error:
        print(f"[SYSTEM ERROR] Failed to log SFT data entry: {log_error}")


def maybe_save_output(query: str, response: str) -> bool:
    if not any(word in query.lower() for word in SAVE_KEYWORDS):
        return False

    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{response}\n\n"
        with open(SAVE_OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write(formatted_text)
        return True
    except Exception as e:
        print(f"[SYSTEM] Failed to write file natively: {e}")
        return False

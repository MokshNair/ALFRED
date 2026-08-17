import subprocess
import sys

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver

from agent import build_agent_executor, build_llm, log_sft_turn, maybe_save_output

subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
load_dotenv()
USE_LOCAL = "--local" in sys.argv

if USE_LOCAL:
    print("[ALFRED] Running on local model (Qwen 2.5 14B)")
else:
    print("[ALFRED] Running on OpenRouter")

llm = build_llm(USE_LOCAL)

# Initializing a local SQLite database to store ALFRED's memory
with SqliteSaver.from_conn_string("ALFRED_memory.db") as memory:

    agent_executor = build_agent_executor(llm, memory)

    # Conversations are saved through "Thread IDs"
    config = {"configurable": {"thread_id": "session_1"}}

    print("\n[SYSTEM] ALFRED is online. Type 'exit' or 'quit' to stop.")

    while True:
        query = input("\nYou: ")

        if query.lower() in ["exit", "quit"]:
            print("Shutting down ALFRED...")
            break

        print("ALFRED is executing...\n")

        # Invocation structure for LangGraph (streaming through nodes)
        try:
            final_state = agent_executor.invoke(
                {"messages": [("user", query)]},
                config=config
            )
            raw_text_output = final_state["messages"][-1].content
        except Exception as e:
            print(f"\n[ALFRED] I encountered an error: {e}")
            continue

        # Automated SFT Data Collection Engine
        log_sft_turn(query, raw_text_output)

        # Native Python File Saving (No LLM Hallucinations)
        if maybe_save_output(query, raw_text_output):
            print(f"[SYSTEM] Successfully hard-saved complete research to research_output.txt")
        else:
            print("\nALFRED:")
            print(raw_text_output)

import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from tools import search_tool, wiki_tool, knowledge_tool
from langgraph.checkpoint.sqlite import SqliteSaver
import datetime

load_dotenv()

# Grab the key securely from the environment
openrouter_key = os.getenv("OPENROUTER_API_KEY")

if not openrouter_key:
    raise ValueError("CRITICAL: OPENROUTER_API_KEY is missing from your .env file!")

# Initializing our Llama3 Model (the Brain)
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_key,
    model="meta-llama/llama-3.3-70b-instruct",
    temperature=0
)

# Defining the tools
tools = [search_tool, wiki_tool, knowledge_tool]


# LangGraph Agent
system_prompt = """
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

# Initializing a local SQLite database to store ALFRED's memory
with SqliteSaver.from_conn_string("ALFRED_memory.db") as memory:
    
    agent_executor = create_agent(
        model=llm, 
        tools=tools, 
        system_prompt=system_prompt,
        checkpointer=memory
    )

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
        try:
            sft_log_file = "sft_training_data.jsonl"
            
            # Structure according to standard ChatML format for SFT
            sft_turn = {
                "messages": [
                    {"role": "system", "content": system_prompt.strip()},
                    {"role": "user", "content": query.strip()},
                    {"role": "assistant", "content": raw_text_output.strip()}
                ]
            }
            
            # Append directly as a new JSON line
            with open(sft_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(sft_turn) + "\n")
                
        except Exception as log_error:
            print(f"[SYSTEM ERROR] Failed to log SFT data entry: {log_error}")
        
        

        # Native Python File Saving (No LLM Hallucinations) 
        if any(word in query.lower() for word in ["save", "write to file", "store", "export"]):
            filename = "research_output.txt"
            try:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{raw_text_output}\n\n"
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(formatted_text)
                print(f"[SYSTEM] Successfully hard-saved complete research to {filename}")
            except Exception as e:
                print(f"[SYSTEM] Failed to write file natively: {e}")
        else:
            print("\nALFRED:")
            print(raw_text_output)
            
            



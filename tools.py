from pydantic import BaseModel, Field
from ddgs import DDGS
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import tool
from memory import sync_knowledge_base, search_knowledge_base
from datetime import datetime


# Sync DB on startup
_vectorstore = sync_knowledge_base()

class KnowledgeBaseInput(BaseModel):
    query: str = Field(description="What to search for in personal notes, PDFs, PPTX, and saved documents.")

@tool("search_knowledge_base", args_schema=KnowledgeBaseInput)
def knowledge_tool(query: str) -> str:
    """Searches ALFRED's personal knowledge base — notes, PDFs, presentations, and saved documents belonging to the user."""
    results = search_knowledge_base(_vectorstore, query)
    if not results:
        return "No relevant information found in personal knowledge base."
    return f"From personal knowledge base:\n\n{results}"



# ==========================================
# 1. THE SAVE FILE TOOL 
# Replaced with native Python file handling in main.py to eliminate LLM hallucination bugs
# ==========================================
# class SaveFileInput(BaseModel):
#     data: str = Field(
#         description="The complete, multi-paragraph research text containing all gathered facts to be written to the file."
#     )
#     filename: str = Field(
#         description="The exact name of the file, ending in .txt"
#     )
    
# @tool("save_text_to_file", description="Saves compiled, detailed research summaries to a local text file. CRITICAL: Never call this tool before running a search.", args_schema=SaveFileInput)
# def save_tool(data: str, filename: str) -> str:
#     try:
#         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

#         with open(filename, "w", encoding="utf-8") as f:
#             f.write(formatted_text)
        
#         return f"Success: Data successfully saved to {filename}:\n\n{data}"
#     except Exception as e:
#         return f"Failed to save file due to error: {str(e)}"

# ==========================================
# 2. THE DUCKDUCKGO SEARCH TOOL
# ==========================================
class SearchInput(BaseModel):
    query: str = Field(description="The clear text search query to look up on the live internet.")

@tool("search", args_schema=SearchInput)
def search_tool(query: str) -> str:
    """Searches the live web for real-time information, local listings, and current events."""
    try:
        results = DDGS().text(query, max_results=3)
        if not results:
            return f"No live search results found for: {query}"
            
        clean_results = []
        for item in results:
            clean_results.append(
                f"Title: {item.get('title', 'N/A')}\n"
                f"Snippet: {item.get('body', 'N/A')}\n"
                f"URL: {item.get('href', 'N/A')}"
            )
        return "\n\n---\n\n".join(clean_results)
    except Exception as e:
        return f"Search tracking failed with error: {str(e)}"


# ==========================================
# 3. THE FIXED WIKIPEDIA TOOL
# ==========================================
class WikiInput(BaseModel):
    query: str = Field(description="The specific historical, geographic, or academic term to lookup on Wikipedia.")

# Expanded doc_content_chars_max to 3000 tokens so the agent gets real substance
wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=3000)

@tool("wikipedia", args_schema=WikiInput)
def wiki_tool(query: str) -> str:
    """Queries Wikipedia for verified academic, historical, geographic, or structural documentation."""
    try:
        results = wiki_wrapper.run(query)
        if not results:
            return f"No Wikipedia pages found matching: {query}"
        return results
    except Exception as e:
        return f"Wikipedia extraction failed: {str(e)}"
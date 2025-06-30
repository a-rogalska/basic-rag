import os

from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode

from basic_rag.doc_stores import vector_store

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if TAVILY_API_KEY:
    tavily_search_tool = TavilySearch(
        max_results=3,
        topic="general",
    )


@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query from the uploaded documents."""
    retrieved_docs = vector_store.similarity_search(query, k=4)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


def init_tools() -> tuple[list, ToolNode]:
    tools = [retrieve]
    if TAVILY_API_KEY:
        tools.append(tavily_search_tool)
    tool_node = ToolNode(tools)
    return tools, tool_node


tools, tool_node = init_tools()

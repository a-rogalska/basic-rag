from langchain_core.messages import SystemMessage
from langchain_core.messages import trim_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import tools_condition

from basic_rag.models import llm
from basic_rag.tools import tool_node
from basic_rag.tools import tools

graph_builder = StateGraph(MessagesState)
trimmer = trim_messages(strategy="last", max_tokens=2, token_counter=len)


def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools(tools)
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def generate(state: MessagesState):
    """Generate answer."""
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    trimmed_messages = trimmer.invoke(conversation_messages)
    prompt = [SystemMessage(system_message_content)] + trimmed_messages

    response = llm.invoke(prompt)
    return {"messages": [response]}


def init_graph() -> CompiledStateGraph:
    graph_builder = StateGraph(MessagesState)

    graph_builder.add_node(query_or_respond)
    graph_builder.add_node(tool_node)
    graph_builder.add_node(generate)

    graph_builder.set_entry_point("query_or_respond")
    graph_builder.add_conditional_edges(
        "query_or_respond",
        tools_condition,
        {END: END, "tools": "tools"},
    )
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)

    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)
    return graph


graph = init_graph()

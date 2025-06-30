import streamlit as st
import torch

torch.classes.__path__ = []

st.set_page_config(
    page_title="Basic RAG",
    page_icon="👋",
)

st.write("# Welcome to a Basic RAG! 👋")

st.markdown(
    """
    Basic RAG is a simple example of an agentic Retrieval-Augmented Generation 
    (RAG) application using Streamlit and LangChain. It allows you to 
    query a knowledge base and get answers based on the retrieved documents 
    and perform web-based search using Tavily.
    """
)

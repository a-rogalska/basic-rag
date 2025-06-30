# Basic RAG
This is a simple agentic Retrieval-Augmented Generation (RAG) system that can be used as a template for the projects. It is built in LangChain and LangGraph with Chroma and OpenAI support. The user interface is developed in Streamlit.

![](/files/ui.png)

## Current Features
* Upload and manage documents in the local Chroma vectorstore
* Query an LLM based on uploaded documents
* Query an LLM to perform web search with Tavily (optional)

## Prerequisites
Ensure you have installed:

- Python 3.13
- uv

## Installation
1. Clone this repository
```bash
git clone https://github.com/a-rogalska/basic-rag.git
cd basic-rag
```

2. Add necessary environment variables to the .env file

    LangSmith and Tavily env variables are optional

3. Install packages
```bash
uv sync
```

4. Run streamlit app
```bash
#Windows
.\.venv\Scripts\activate
# execute this in root of the project
streamlit run .\src\Welcome.py
```

## Planned Features
* Voice support with local models
* Local LLMs support with Ollama
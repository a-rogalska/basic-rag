import os

from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

llm = init_chat_model(OPENAI_MODEL, model_provider="openai")

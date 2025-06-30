from abc import ABC
from abc import abstractmethod

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from basic_rag.models import embeddings


class DocumentStore(ABC):
    """Abstract base class for document stores."""

    @abstractmethod
    def add_document(self, id, document):
        """Add document to the store."""
        pass

    @abstractmethod
    def delete_document(self, id):
        """Delete document from the store."""
        pass


class ChromaStore(DocumentStore):
    """Chroma document store implementation."""
    collection_name = "documents"

    def __init__(self, embedding_function) -> None:
        _persistent_client = chromadb.PersistentClient()
        self.collection = _persistent_client.get_or_create_collection(self.collection_name)
        self.vector_store = Chroma(
            client=_persistent_client,
            collection_name=self.collection_name,
            embedding_function=embedding_function
        )

    def add_document(self, id, document_content):
        """Add document to the collection."""
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        document = Document(page_content=document_content, metadata={"filename": id})
        all_splits = text_splitter.split_documents([document])
        self.vector_store.add_documents(documents=all_splits)

    def delete_document(self, id):
        """Delete document from the collection."""
        self.vector_store.delete(ids=id)

    def get_documents(self):
        """Get all documents from the collection."""
        return self.vector_store.get()


chroma_store = ChromaStore(embedding_function=embeddings)
vector_store = chroma_store.vector_store

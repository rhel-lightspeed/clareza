"""Embed data in the vector database."""

from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
)

from clareza.infra import get_embedding_model, get_vector_store


def clear_documents() -> None:
    """Clear all documents from the vector database."""
    vector_store = get_vector_store()
    vector_store.clear()


def embed_documents(docs: list) -> None:  # pragma: no cover
    """Embed documents in the vector database."""
    documents = [Document(extra_info=doc.metadata, text=doc.text) for doc in docs]
    vector_store = get_vector_store()
    Settings.embed_model = get_embedding_model()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex.from_documents(documents, storage_context=storage_context)

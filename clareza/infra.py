"""Infrastructure for performing AI-related functions."""

from llama_index.core import Settings
from llama_index.embeddings.ibm import WatsonxEmbeddings
from llama_index.llms.ibm import WatsonxLLM
from llama_index.vector_stores.postgres import PGVectorStore

from clareza.settings import Settings as LocalSettings


def get_embedding_model() -> WatsonxEmbeddings:
    """Get the default embedding model."""
    local_settings = LocalSettings()  # type: ignore[call-arg]
    watsonx_embedding = WatsonxEmbeddings(
        model_id=local_settings.EMBED_MODEL,
        url=local_settings.WATSONX_URL,
        project_id=local_settings.WATSONX_PROJECT_ID,
    )

    # Set the embedding model in the llamaindex Settings (otherwise OpenAI is used).
    Settings.embed_model = watsonx_embedding

    return watsonx_embedding


def get_llm_model() -> WatsonxLLM:
    """Get the LLM model."""
    watsonx_llm = WatsonxLLM(
        model_id="ibm/granite-3-2b-instruct",
        url=LocalSettings.WATSONX_URL,
        project_id=LocalSettings.WATSONX_PROJECT_ID,
        temperature=LocalSettings.LLM_TEMPERATURE,
        max_new_tokens=LocalSettings.LLM_MAX_NEW_TOKENS,
    )

    # Set the LLM model in the llamaindex Settings (otherwise OpenAI is used).
    Settings.llm = watsonx_llm
    return watsonx_llm


def get_vector_store() -> PGVectorStore:
    """Get the vector store."""
    local_settings = LocalSettings()  # type: ignore[call-arg]
    return PGVectorStore.from_params(
        host=local_settings.DB_HOST,
        password=local_settings.DB_PASSWORD,
        port=local_settings.DB_PORT,
        user=local_settings.DB_USER,
        database=local_settings.DB_NAME,
        table_name=local_settings.DB_TABLE_NAME,
        embed_dim=local_settings.DB_EMBED_DIM,
        hybrid_search=True,
        text_search_config="english",
        perform_setup=True,
    )

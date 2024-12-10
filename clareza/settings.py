from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    WATSONX_APIKEY: str
    WATSONX_PROJECT_ID: str
    WATSONX_URL: str = "https://us-south.ml.cloud.ibm.com"
    EMBED_MODEL: str = "ibm/slate-30m-english-rtrvr-v2"
    LLM_MODEL: str = "ibm/granite-3-2b-instruct"
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_NEW_TOKENS: int = 2048
    DB_HOST: str = "127.0.0.1"
    DB_PORT: str = "5432"
    DB_USER: str = "postgres"
    DB_PASSWORD: str = "secrete"
    DB_NAME: str = "vectorsandthings"
    DB_TABLE_NAME: str = "clareza"
    DB_EMBED_DIM: int = 384


settings = Settings()  # type: ignore[call-arg]

from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel

from clareza.embed import clear_documents, embed_documents

app = FastAPI()


class DocumentToEmbed(BaseModel):
    metadata: dict
    text: str


@app.get("/")
def read_root() -> ORJSONResponse:
    resp = {"Hello": "World"}
    return ORJSONResponse({"data": resp})


@app.post("/embed")
async def add_documents_to_vector_store(docs: list[DocumentToEmbed]) -> ORJSONResponse:
    embed_documents(docs)
    return ORJSONResponse({"data": "Success"})


@app.delete("/embed")
async def clear_vector_store() -> ORJSONResponse:
    clear_documents()
    return ORJSONResponse({"data": "Success"})

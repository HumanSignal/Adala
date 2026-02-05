import chromadb
import hashlib
from .base import Memory
from uuid import uuid4
from pydantic import BaseModel, Field, model_validator
from typing import Any, List, Dict, Optional

from openai import OpenAI


class OpenAIEmbeddingFunction:
    """
    ChromaDB embedding function using the OpenAI Python SDK (v1+).

    ChromaDB's built-in `embedding_functions.OpenAIEmbeddingFunction` relies on the
    legacy `openai.Embedding` API which was removed in openai>=1.0.
    """

    def __init__(
        self, *, model_name: str, api_key: str, base_url: Optional[str] = None
    ):
        self._model_name = model_name
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def __call__(self, input: List[str]) -> List[List[float]]:
        resp = self._client.embeddings.create(model=self._model_name, input=input)
        # Keep ordering stable
        return [item.embedding for item in resp.data]


class VectorDBMemory(Memory):
    """
    Memory backed by a vector database.
    """

    db_name: str = ""
    openai_api_key: str
    openai_embedding_model: str = "text-embedding-ada-002"
    _client = None
    _collection = None
    _embedding_function = None

    @model_validator(mode="after")
    def init_database(self):
        self._client = chromadb.Client()
        self._embedding_function = OpenAIEmbeddingFunction(
            model_name=self.openai_embedding_model, api_key=self.openai_api_key
        )
        self._collection = self._client.get_or_create_collection(
            name=self.db_name, embedding_function=self._embedding_function
        )
        return self

    def create_unique_id(self, string):
        return hashlib.md5(string.encode()).hexdigest()

    def remember(self, observation: str, data: Any):
        self.remember_many([observation], [data])

    def remember_many(self, observations: List[str], data: List[Dict]):
        # filter None values from each item in `data`
        data = [{k: v for k, v in d.items() if v is not None} for d in data]

        self._collection.add(
            ids=[self.create_unique_id(o) for o in observations],
            documents=observations,
            metadatas=data,
        )

    def retrieve_many(self, observations: List[str], num_results: int = 1) -> List[Any]:
        result = self._collection.query(query_texts=observations, n_results=num_results)
        return result["metadatas"]

    def retrieve(self, observation: str, num_results: int = 1) -> Any:
        return self.retrieve_many([observation], num_results=num_results)[0]

    def clear(self):
        self._client.delete_collection(name=self.db_name)
        self._collection = self._client.create_collection(
            name=self.db_name, embedding_function=self._embedding_function
        )

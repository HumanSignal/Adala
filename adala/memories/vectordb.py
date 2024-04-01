import chromadb
import hashlib
from .base import Memory
from uuid import uuid4
from pydantic import BaseModel, Field, model_validator
from chromadb.utils import embedding_functions
from typing import Any, List, Dict


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
        self._embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            model_name=self.openai_embedding_model,
            api_key=self.openai_api_key
        )
        self._collection = self._client.get_or_create_collection(
            name=self.db_name, embedding_function=self._embedding_function
        )

    def create_unique_id(self, string):
        return hashlib.md5(string.encode()).hexdigest()

    def remember(self, observation: str, data: Any):
        self.remember_many([observation], [data])

    def remember_many(self, observations: List[str], data: List[Dict]):
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

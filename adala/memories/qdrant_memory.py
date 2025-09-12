from typing import Any, List, Dict, Optional
import uuid
from pydantic import Field, model_validator

from .base import Memory

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    import openai

    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False


class QdrantMemory(Memory):
    """
    Memory backed by [Qdrant](https://qdrant.tech/).
    """

    model_config = {"arbitrary_types_allowed": True}

    collection_name: str = Field(..., description="Name of the Qdrant collection")
    openai_api_key: str = Field(..., description="OpenAI API key for embeddings")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small", description="OpenAI embedding model"
    )
    qdrant_url: Optional[str] = Field(
        default=None, description="Qdrant server URL"
    )
    qdrant_api_key: Optional[str] = Field(
        default=None, description="Qdrant API key for remote instances"
    )
    qdrant_client: Optional[QdrantClient] = Field(
        default=None, description="Pre-configured QdrantClient instance"
    )
    dimension: int = Field(default=1536, description="Vector dimension size")
    distance_metric: str = Field(
        default="Cosine", description="Distance metric for similarity search"
    )

    _client: Optional[QdrantClient] = None
    _openai_client: Optional[openai.OpenAI] = None

    @model_validator(mode="after")
    def init_database(self):
        if not QDRANT_AVAILABLE:
            raise ImportError(
                "Qdrant dependencies not available. "
                "Please install with: pip install qdrant-client openai"
            )

        if self.qdrant_client is not None and (
            self.qdrant_url is not None or self.qdrant_api_key is not None
        ):
            raise ValueError(
                "Cannot specify both 'qdrant_client' and 'qdrant_url'/'qdrant_api_key'. "
                "Use either a pre-configured QdrantClient or URL-based configuration, not both."
            )

        if self.qdrant_client is not None:
            self._client = self.qdrant_client
        elif self.qdrant_url:
            self._client = QdrantClient(
                url=self.qdrant_url, api_key=self.qdrant_api_key
            )
        else:
            raise ValueError(
                "No Qdrant configuration provided. Please specify either 'qdrant_client' "
                "or 'qdrant_url' to configure the Qdrant connection."
            )

        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required but not provided")
        self._openai_client = openai.OpenAI(api_key=self.openai_api_key)

        if not self._client.collection_exists(self.collection_name):
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.dimension, distance=self._get_distance_metric()
                ),
            )

        return self

    def _generate_uuid(self, string: str) -> str:
        return uuid.uuid5(uuid.NAMESPACE_URL, string).hex

    def _get_distance_metric(self) -> Distance:
        distance_map = {
            "Cosine": Distance.COSINE,
            "Dot": Distance.DOT,
            "Euclidean": Distance.EUCLID,
            "Manhattan": Distance.MANHATTAN,
        }
        return distance_map.get(self.distance_metric, Distance.COSINE)

    def _get_embedding(self, text: str) -> List[float]:
        response = self._openai_client.embeddings.create(
            model=self.openai_embedding_model, input=text
        )
        return response.data[0].embedding

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        response = self._openai_client.embeddings.create(
            model=self.openai_embedding_model, input=texts
        )
        return [data.embedding for data in response.data]

    def remember(self, observation: str, data: Any):
        """Store a single observation with its associated data."""
        self.remember_many([observation], [data])

    def remember_many(self, observations: List[str], data: List[Dict]):
        """Store multiple observations with their associated data."""

        data = [{k: v for k, v in d.items() if v is not None} for d in data]

        embeddings = self._get_embeddings(observations)

        points = []
        for obs, embedding, metadata in zip(observations, embeddings, data):
            point_id = self._generate_uuid(obs)
            points.append(
                PointStruct(
                    id=point_id, vector=embedding, payload={"text": obs, **metadata}
                )
            )

        self._client.upsert(collection_name=self.collection_name, points=points)

    def retrieve_many(self, observations: List[str], num_results: int = 1) -> List[Any]:
        """Retrieve similar observations for multiple queries."""
        results = []

        for observation in observations:
            query_embedding = self._get_embedding(observation)

            search_results = self._client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=num_results,
                with_payload=True,
            ).points

            metadatas = []
            for result in search_results:
                payload = result.payload.copy()

                payload.pop("text", None)
                metadatas.append(payload)

            results.append(metadatas)

        return results

    def retrieve(self, observation: str, num_results: int = 1) -> Any:
        """Retrieve similar observations for a single query."""
        return self.retrieve_many([observation], num_results=num_results)[0]

    def clear(self):
        """Clear all data from the collection."""

        if self._client.collection_exists(self.collection_name):
            self._client.delete_collection(self.collection_name)

        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.dimension, distance=self._get_distance_metric()
            ),
        )

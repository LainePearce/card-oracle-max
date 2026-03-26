"""Qdrant collection creation and configuration.

Creates the 'cards' collection with named vectors (image 512-dim, specifics 384-dim),
INT8 scalar quantization, and on-disk HNSW. Supports single-node test configuration
via QDRANT_SINGLE_NODE env var.

See CLAUDE.md Section 4.1 for canonical configuration.
"""

import os

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    VectorParams,
)

COLLECTION_NAME = "cards"

# --- Production configuration (3-node cluster) ---

COLLECTION_CONFIG = {
    "collection_name": COLLECTION_NAME,
    "vectors_config": {
        "image": VectorParams(
            size=512,
            distance=Distance.COSINE,
            on_disk=True,
            hnsw_config=HnswConfigDiff(
                m=16,
                ef_construct=200,
                on_disk=True,
            ),
        ),
        "specifics": VectorParams(
            size=384,
            distance=Distance.COSINE,
            on_disk=True,
            hnsw_config=HnswConfigDiff(
                m=16,
                ef_construct=200,
                on_disk=True,
            ),
        ),
    },
    "quantization_config": ScalarQuantization(
        scalar=ScalarQuantizationConfig(
            type=ScalarType.INT8,
            quantile=0.99,
            always_ram=True,
        ),
    ),
    "optimizers_config": OptimizersConfigDiff(
        indexing_threshold=50_000,
        memmap_threshold=50_000,
    ),
    "on_disk_payload": True,
    "shard_number": 6,
    "replication_factor": 2,
    "write_consistency_factor": 1,
}

# --- Single-node test overrides ---
# Applied when QDRANT_SINGLE_NODE=true — reduces shards and disables replication
# so the collection works on a single EC2 instance or local Docker.

SINGLE_NODE_OVERRIDES = {
    "shard_number": 1,
    "replication_factor": 1,
    "write_consistency_factor": 1,
}


def is_single_node() -> bool:
    return os.environ.get("QDRANT_SINGLE_NODE", "false").lower() in ("true", "1", "yes")


def get_collection_config() -> dict:
    """Return collection config, applying single-node overrides if configured."""
    config = {**COLLECTION_CONFIG}
    if is_single_node():
        config.update(SINGLE_NODE_OVERRIDES)
    return config


def create_collection(client: QdrantClient | None = None) -> None:
    """Create the cards collection with the appropriate configuration.

    If no client is provided, connects using QDRANT_HOST, QDRANT_PORT,
    and QDRANT_API_KEY environment variables.
    """
    if client is None:
        host = os.environ.get("QDRANT_HOST", "localhost")
        port = int(os.environ.get("QDRANT_PORT", 6333))
        api_key = os.environ.get("QDRANT_API_KEY")
        use_grpc = os.environ.get("QDRANT_USE_GRPC", "false").lower() == "true"

        # Use explicit URL to avoid the client assuming TLS on remote hosts
        client = QdrantClient(
            url=f"http://{host}:{port}",
            api_key=api_key,
            prefer_grpc=use_grpc,
        )

    config = get_collection_config()

    # Check if collection already exists
    collections = client.get_collections().collections
    if any(c.name == COLLECTION_NAME for c in collections):
        print(f"Collection '{COLLECTION_NAME}' already exists — skipping creation.")
        return

    client.create_collection(**config)

    mode = "single-node" if is_single_node() else "cluster"
    shards = config["shard_number"]
    replicas = config["replication_factor"]
    print(
        f"Collection '{COLLECTION_NAME}' created ({mode}: "
        f"{shards} shard(s), replication_factor={replicas})"
    )


if __name__ == "__main__":
    create_collection()

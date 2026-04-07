"""Bulk writer for the new self-managed Elasticsearch cluster.

Writes documents in batches using the _bulk API.
No vector fields — Qdrant handles all vector search.

Uses the elasticsearch-py 8.x client (not opensearch-py).
The Elasticsearch 8.x HTTP API is identical to OpenSearch for the operations
we use: _bulk, _index_template, _cat/indices, _cluster/health.
"""

from __future__ import annotations

import os
import time
from typing import Generator

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk as es_bulk
from loguru import logger


BULK_BATCH_SIZE = 1000
BULK_CHUNK_SIZE = 500   # docs per _bulk request (within a batch)
MAX_RETRIES     = 3
RETRY_DELAY     = 5     # seconds


def get_es_client() -> Elasticsearch:
    """
    Connect to the NEW self-managed Elasticsearch cluster.
    Uses OPENSEARCH_DOCS_* env vars — same names as before for compatibility
    with existing .env files and Terraform user_data scripts.

    Security (TLS, auth) is disabled at the cluster level for VPC-internal use.
    If OPENSEARCH_DOCS_USER and OPENSEARCH_DOCS_PASSWORD are set, basic auth
    is used even without TLS (valid when xpack.security.enabled=true but
    xpack.security.http.ssl.enabled=false).
    """
    host     = os.environ["OPENSEARCH_DOCS_HOST"]
    port     = int(os.environ.get("OPENSEARCH_DOCS_PORT", 9200))
    use_ssl  = os.environ.get("OPENSEARCH_DOCS_USE_SSL", "false").lower() == "true"
    user     = os.environ.get("OPENSEARCH_DOCS_USER", "")
    password = os.environ.get("OPENSEARCH_DOCS_PASSWORD", "")

    scheme = "https" if use_ssl else "http"
    hosts  = [f"{scheme}://{host}:{port}"]

    kwargs: dict = {
        "hosts":         hosts,
        "retry_on_timeout": True,
        "max_retries":   MAX_RETRIES,
        "request_timeout": 60,
    }

    if user and password:
        kwargs["basic_auth"] = (user, password)

    if not use_ssl:
        kwargs["verify_certs"]  = False
        kwargs["ssl_show_warn"] = False

    return Elasticsearch(**kwargs)


# Keep the old name as an alias so existing imports don't break
get_new_os_client = get_es_client


def bulk_index(
    client: Elasticsearch,
    documents: list[tuple[str, str, dict]],
) -> dict:
    """
    Bulk index a batch of documents into the Elasticsearch cluster.

    Args:
        client: Elasticsearch client for the new cluster
        documents: list of (index_name, doc_id, doc_body) tuples

    Returns:
        dict with 'indexed', 'errors' counts
    """
    if not documents:
        return {"indexed": 0, "errors": 0}

    actions = []
    for index_name, doc_id, doc_body in documents:
        actions.append({
            "_index": index_name,
            "_id":    doc_id,
            "_source": doc_body,
        })

    stats = {"indexed": 0, "errors": 0}

    try:
        success, errors = es_bulk(
            client,
            actions,
            chunk_size=BULK_CHUNK_SIZE,
            max_retries=MAX_RETRIES,
            raise_on_error=False,
            raise_on_exception=False,
        )
        stats["indexed"] = success
        if errors:
            stats["errors"] = len(errors)
            for err in errors[:5]:
                logger.error("Bulk index error: {}", err)
    except Exception as e:
        logger.error("Bulk index failed: {}", e)
        stats["errors"] = len(documents)

    return stats


def ensure_index_exists(client: Elasticsearch, index_name: str) -> None:
    """
    Check if index exists; if not, it will be auto-created by the index template.
    This is a no-op if templates are correctly applied — the index is created
    on first write with the template mapping.
    """
    if not client.indices.exists(index=index_name):
        logger.debug("Index {} will be auto-created by template on first write", index_name)


def get_index_doc_counts(client: Elasticsearch, pattern: str = "*") -> dict[str, int]:
    """Get document counts per index for validation/monitoring."""
    response = client.cat.indices(index=pattern, format="json", h="index,docs.count")
    return {
        row["index"]: int(row.get("docs.count") or 0)
        for row in response
        if not row["index"].startswith(".")
    }

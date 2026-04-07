"""OpenSearch index mapping for the new self-managed cluster.

No KNN/vector fields — Qdrant handles all vector search.
This mapping covers document storage, BM25 text search, keyword filters,
aggregations, and mget enrichment.

Mirrors the extant managed cluster mapping minus imageVector/textVector.
"""

from __future__ import annotations

import os
from loguru import logger
from opensearchpy import OpenSearch


# ── Index Settings (no KNN) ─────────────────────────────────────

INDEX_SETTINGS = {
    "index": {
        # KNN explicitly disabled — vectors live in Qdrant
        "knn": False,
        "number_of_shards": 1,
        "number_of_replicas": 1,
        "refresh_interval": "30s",
    },
    "analysis": {
        "normalizer": {
            "lowercase_normalizer": {
                "type": "custom",
                "filter": ["lowercase", "asciifolding"],
            }
        }
    },
}


# ── Field Mappings ───────────────────────────────────────────────

FIELD_MAPPINGS = {
    "properties": {
        # ── Primary identity ──────────────────────────────────
        "id": {"type": "long"},
        "itemId": {
            "type": "text",
            "fields": {"keyword": {"type": "keyword"}},
        },
        "source": {"type": "keyword"},
        "globalId": {"type": "keyword"},

        # ── Title — full-text + autocomplete + exact match ────
        "title": {
            "type": "text",
            "fields": {
                "keyword": {"type": "keyword", "ignore_above": 256},
                "suggest": {"type": "search_as_you_type"},
            },
        },

        # ── URLs ──────────────────────────────────────────────
        "galleryURL": {"type": "keyword", "ignore_above": 2048},
        "itemURL": {"type": "keyword", "ignore_above": 2048},

        # ── Pricing / sale ────────────────────────────────────
        "saleType": {"type": "keyword"},
        "currentPrice": {"type": "double"},
        "currentPriceCurrency": {"type": "keyword"},
        "salePrice": {"type": "double"},
        "salePriceCurrency": {"type": "keyword"},
        "shippingServiceCost": {"type": "double"},
        "BestOfferPrice": {"type": "double"},
        "BestOfferCurrency": {"type": "keyword"},
        "bidCount": {"type": "integer"},

        # ── Dates ─────────────────────────────────────────────
        "endTime": {"type": "date", "format": "yyyy-MM-dd HH:mm:ss||epoch_millis"},
        "startTime": {"type": "date", "format": "yyyy-MM-dd HH:mm:ss||epoch_millis"},

        # ── Misc ──────────────────────────────────────────────
        "cloud": {"type": "integer"},

        # ── itemSpecifics (all keyword with lowercase normalizer) ─
        "itemSpecifics": {
            "properties": {
                "brand": {
                    "type": "keyword",
                    "normalizer": "lowercase_normalizer",
                },
                "player": {
                    "type": "keyword",
                    "normalizer": "lowercase_normalizer",
                },
                "genre": {
                    "type": "keyword",
                    "normalizer": "lowercase_normalizer",
                },
                "country": {
                    "type": "keyword",
                    "normalizer": "lowercase_normalizer",
                },
                "set": {
                    "type": "keyword",
                    "normalizer": "lowercase_normalizer",
                },
                "cardNumber": {
                    "type": "keyword",
                    "normalizer": "lowercase_normalizer",
                },
                "subset": {
                    "type": "keyword",
                    "normalizer": "lowercase_normalizer",
                },
                "parallel": {
                    "type": "keyword",
                    "normalizer": "lowercase_normalizer",
                },
                "serialNumber": {
                    "type": "keyword",
                    "normalizer": "lowercase_normalizer",
                },
                "year": {"type": "integer"},
                "graded": {"type": "boolean"},
                "grader": {
                    "type": "keyword",
                    "normalizer": "lowercase_normalizer",
                },
                "grade": {
                    "type": "keyword",
                    "normalizer": "lowercase_normalizer",
                },
                "type": {
                    "type": "keyword",
                    "normalizer": "lowercase_normalizer",
                },
                "autographed": {"type": "boolean"},
                "team": {
                    "type": "keyword",
                    "normalizer": "lowercase_normalizer",
                },
            }
        },
    }
}


# ── Index Templates ──────────────────────────────────────────────

# Each marketplace has a distinct index naming pattern.
# Templates auto-apply the correct mapping when a new index is created.

INDEX_TEMPLATES = {
    "ebay-dated": {
        "index_patterns": ["????-??-??"],
        "priority": 100,
    },
    "goldin": {
        "index_patterns": ["????-gold"],
        "priority": 90,
    },
    "fanatics-pwcc": {
        "index_patterns": ["????-??-pwcc"],
        "priority": 90,
    },
    "pristine": {
        "index_patterns": ["????-??-pris"],
        "priority": 90,
    },
    "myslabs": {
        "index_patterns": ["????-ms"],
        "priority": 90,
    },
    "heritage": {
        "index_patterns": ["????-heri"],
        "priority": 90,
    },
    "cardhobby": {
        "index_patterns": ["????-cardhobby"],
        "priority": 90,
    },
    "rea": {
        "index_patterns": ["????-rea"],
        "priority": 90,
    },
    "veriswap": {
        "index_patterns": ["????-veriswap"],
        "priority": 90,
    },
}


def create_index_templates(client: OpenSearch) -> None:
    """
    Apply all index templates to the cluster.
    Templates auto-apply the mapping when new indices matching the pattern are created.
    Uses composable index templates (OpenSearch 2.x).
    """
    # First create the component template with shared settings + mappings
    component_body = {
        "template": {
            "settings": INDEX_SETTINGS,
            "mappings": FIELD_MAPPINGS,
        },
    }
    client.cluster.put_component_template(
        name="card-oracle-base",
        body=component_body,
    )
    logger.info("Created component template: card-oracle-base")

    # Create composable index templates referencing the component
    for name, config in INDEX_TEMPLATES.items():
        template_body = {
            "index_patterns": config["index_patterns"],
            "priority": config["priority"],
            "composed_of": ["card-oracle-base"],
        }
        client.indices.put_index_template(
            name=f"card-oracle-{name}",
            body=template_body,
        )
        logger.info(
            "Created index template: card-oracle-{} (patterns: {})",
            name, config["index_patterns"],
        )


def create_index(client: OpenSearch, index_name: str) -> None:
    """
    Explicitly create a single index with the mapping.
    Normally not needed if templates are applied — this is for manual creation.
    """
    if client.indices.exists(index=index_name):
        logger.info("Index {} already exists, skipping", index_name)
        return

    body = {
        "settings": INDEX_SETTINGS,
        "mappings": FIELD_MAPPINGS,
    }
    client.indices.create(index=index_name, body=body)
    logger.info("Created index: {}", index_name)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    from src.ingestion.opensearch_reader import get_opensearch_client

    # Connect to the NEW self-managed cluster (not the extant managed one)
    # Set OPENSEARCH_HOST to the new cluster endpoint before running
    client = get_opensearch_client()

    logger.info("Applying index templates to cluster at {}", os.environ.get("OPENSEARCH_HOST"))
    create_index_templates(client)
    logger.info("All templates applied successfully")

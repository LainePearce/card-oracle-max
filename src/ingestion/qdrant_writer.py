"""Bulk upsert writer for Qdrant.

Extracts payload from OpenSearch documents, normalises field values
(lowercase + strip to match OpenSearch lowercase_normalizer), and
upserts points in batches.
"""

UPSERT_BATCH_SIZE = 1_000
UPSERT_WAIT = False
CHECKPOINT_EVERY = 10_000

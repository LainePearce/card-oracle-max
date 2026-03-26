"""S3 vector persistence layer.

S3 is the primary durable store for all generated vectors.
Vectors are written to S3 BEFORE being loaded into Qdrant.
Qdrant is a queryable index built from S3, not the source of truth.

Supports: repopulation, model versioning, partial reload, audit.
See CLAUDE.md Section 16 for full specification.
"""

SHARD_SIZE = 100_000
COMPRESSION = "snappy"

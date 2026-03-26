"""Ingestion orchestration: OpenSearch scroll -> embed -> S3 -> Qdrant.

Routes documents through the correct extraction path based on index type
(eBay dated vs marketplace suffix) and development stage.

Stage 1: eBay docs get full embeddings; non-eBay get image vector only.
Stage 2: Non-eBay docs go through specifics inference pipeline.
"""

"""AWS Batch GPU worker entry point for embedding generation.

Reads documents from OpenSearch via scroll, generates CLIP and MiniLM embeddings,
writes vectors to S3, then loads into Qdrant.
"""

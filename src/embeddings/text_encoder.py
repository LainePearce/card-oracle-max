"""MiniLM-L6-v2 itemSpecifics text embedding encoder.

Produces 384-dim vectors from flattened itemSpecifics fields.
CPU-friendly — can run in parallel with GPU image encoding.
"""

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
BATCH_SIZE = 1024

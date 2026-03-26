"""CLIP ViT-L/14 image embedding encoder.

Produces 512-dim L2-normalised vectors from card images via galleryURL.
Uses OpenCLIP with FP16 inference on GPU.
"""

MODEL_NAME = "ViT-L/14"
EMBEDDING_DIM = 512
BATCH_SIZE = 256
USE_FP16 = True
NORMALISE = True

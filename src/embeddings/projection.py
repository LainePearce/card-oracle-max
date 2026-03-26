"""Cross-modal projection: 512-dim CLIP image space -> 384-dim MiniLM specifics space.

Small learned projection trained on sold cards (both vectors known) using cosine
similarity loss. Required for querying Type 3 catalogue cards (no image) via image query.
"""

PROJECTION_MODEL_PATH = "models/image_to_specifics_projection.pt"
IMAGE_DIM = 512
SPECIFICS_DIM = 384

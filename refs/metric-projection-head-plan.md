# Metric Projection Head on CLIP Embeddings — Development Plan

---

## Phase 1: Data Extraction from OpenSearch

**Goal:** Build a corpus of `(os_id, galleryURL, itemSpecifics, identity_key)` tuples where identity determines "same card."

### 1.1 Scroll query filter

Only include documents where:
- `galleryURL` is non-null and non-empty
- All essential fields present and non-blank: `year`, `brand`, `player`, `set`, `cardNumber`
- Source index is `YYYY-MM-DD` (eBay only — itemSpecifics guaranteed present)

```python
REQUIRED_SPECIFICS  = {"year", "brand", "player", "set", "cardNumber"}
DESIRABLE_SPECIFICS = {"subset", "serialNumber", "parallel"}
GRADING_SPECIFICS   = {"graded", "grader", "grade"}
```

For each scrolled document, extract and lowercase all specifics. Discard any document where any essential field is empty string, `"null"`, `"n/a"`, or `"unknown"` after normalisation.

### 1.2 Identity key design

The identity key defines what "same card" means. Two documents with the same key are positives; different keys are negatives.

```
Base key (all datasets):
  (brand, set, cardNumber, year, player)
  → lowercased, stripped, normalised

Extended key (when parallel/subset present):
  base_key + (parallel or "raw", subset or "")
  → used for hard negative generation only
  → two cards with same base_key but different parallel are
    "related but not identical" — useful as hard negatives
```

### 1.3 Three dataset splits

| Dataset | Filter | Identity includes grade? | Size estimate |
|---------|--------|-------------------------|---------------|
| **D1 — Raw** | `graded == false OR graded missing` | No | ~60–70% of corpus |
| **D2 — Graded** | `graded == true`, grader + grade both present | Yes — `(base_key, grader, grade)` | ~25–30% |
| **D3 — Combined** | All items | No — grade ignored | 100% |

D3 is the production training set. D1 and D2 are used for fine-grained evaluation and optional specialist models.

---

## Phase 2: Embedding Extraction

Reuse the existing S3 vector store infrastructure (`src/embeddings/vector_store.py`).

### 2.1 Batch job

Run the existing AWS Batch CLIP job across all qualifying date indices. Each record written to S3 parquet:

```
os_id | qdrant_id | index_name | vector (512-dim) | identity_key | graded | grader | grade | parallel | subset
```

Add `identity_key` as an additional column to the parquet schema for this training run. This avoids re-querying OpenSearch during training.

### 2.2 Deduplication

Multiple sold listings of the same card = multiple images with the same identity key. This is exactly what we want. Do **not** deduplicate. Quantity of listings per card is the source of positives.

Minimum threshold: **retain identity keys that have ≥ 2 images** (need at least one positive pair). Keys with only one image can be used as negatives but not anchors.

Aim for: **≥ 5 images per identity key** for the training split to get meaningful SupCon batches.

---

## Phase 3: Pair and Triplet Construction

### 3.1 Loss choice — Supervised Contrastive (SupCon)

Triplet loss requires explicit triplet mining. SupCon handles multiple positives per anchor naturally, which suits this data (one card → many sold listings). Use **SupCon** as the primary loss:

```
L_SupCon = -1/|P(i)| * Σ_{p∈P(i)} log[ exp(z_i·z_p/τ) / Σ_{a∈A(i)} exp(z_i·z_a/τ) ]

where P(i) = positives for anchor i (same identity_key)
      A(i) = all other items in batch
      τ    = temperature (start at 0.07)
```

### 3.2 Batch construction — online hard negative mining

Do **not** pre-generate pairs. Construct batches with class-balanced sampling:

```
Batch size: 256
Sampling:   N_classes_per_batch × K_samples_per_class
            e.g. 64 identity keys × 4 images each = 256

Within a batch:
  Positives:      any two samples sharing identity_key
  Easy negatives: different brand/player — random
  Hard negatives: same player, different set
                  same set, different cardNumber
                  same base_key, different parallel
                  same base_key, different grade (graded cards)
```

Hard negatives make the model learn the fine distinctions CLIP misses. Mining them online (within the batch) is more efficient than pre-computing.

### 3.3 Hard negative curriculum

Apply a **curriculum**: start with easy negatives for the first 5 epochs, progressively increase the hard negative ratio to 70% by epoch 15. This prevents early collapse.

---

## Phase 4: Model Architecture

Small MLP on top of frozen CLIP. Keeps CLIP's general visual knowledge, adapts the metric space for cards.

```python
class CardMetricHead(nn.Module):
    """
    Projection head trained on top of frozen CLIP ViT-L/14 (512-dim).
    Maps card embeddings into a 256-dim L2-normalised metric space.
    """
    def __init__(self, input_dim=512, hidden_dim=512, output_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return F.normalize(out, dim=-1)   # L2 normalise — required for cosine similarity
```

CLIP is **frozen throughout** — only the projection head weights are trained. This keeps training fast (no backprop through ViT) and prevents catastrophic forgetting.

Output dim of 256 is a deliberate reduction from 512 — forces the head to discard CLIP's generic visual features and retain only card-discriminative ones.

---

## Phase 5: Training Setup

### 5.1 Data splits

```
Train:      80% of identity keys (all images for those keys)
Validation: 10% of identity keys
Test:       10% of identity keys (held out until final evaluation)

Split at identity key level — never split the same key across train/val/test.
This ensures the model is evaluated on genuinely unseen cards.
```

### 5.2 Hyperparameters

```python
BATCH_SIZE     = 256
LEARNING_RATE  = 1e-4
WEIGHT_DECAY   = 1e-2
EPOCHS         = 30
WARMUP_EPOCHS  = 3
TEMPERATURE    = 0.07     # SupCon τ — tune if loss plateaus
OPTIMIZER      = AdamW
SCHEDULER      = CosineAnnealingLR(T_max=30, eta_min=1e-6)
```

### 5.3 Training infrastructure

Run on the existing g5.xlarge GPU instance. CLIP embeddings are pre-extracted to S3, so the training loop only loads 512-dim float32 vectors — no image loading, no GPU-bound preprocessing. Full corpus of 50M embeddings at 512-dim ≈ 100GB on S3, loaded from parquet shards on demand. The projection head is tiny so each forward pass is fast.

Estimated training time at 50M samples, batch size 256, 30 epochs: **~8–12 hours** on g5.xlarge.

---

## Phase 6: Evaluation Metrics

### 6.1 Recall@K on held-out query set

For each test image:
- Query: CLIP embedding → project → nearest neighbours by cosine similarity
- Ground truth: all other images with same identity_key
- Measure: Recall@1, Recall@5, Recall@10

Compare against raw CLIP (no projection) as baseline.

### 6.2 Hard negative discrimination

For each test anchor, report:
- Mean similarity to true positives (same card)
- Mean similarity to hard negatives (same player, different card)
- **Gap** between them — this is the key improvement metric

### 6.3 Grade discrimination (D2 — graded dataset)

For graded cards:
- Can the model distinguish PSA 9 from PSA 10 of the same card?
- Measure: whether `(same_card, same_grade)` scores higher than `(same_card, different_grade)`

### 6.4 Raw vs graded separation

For D3 combined:
- Does a raw card score higher against other raw copies than against graded copies?
- Not a requirement, but indicates whether grading condition bleeds into the similarity space

---

## Phase 7: Integration into Production

### 7.1 S3 model storage

```
s3://{bucket}/models/metric-head/
  v1-clip-vitl14-supcon/
    head.pt            # state_dict of CardMetricHead
    config.json        # input_dim, hidden_dim, output_dim, clip_model
    eval_results.json
```

### 7.2 GPU worker integration

In `tools/gpu_worker_server.py`, load the head alongside CLIP at startup:

```python
_metric_head = None

def get_metric_head():
    global _metric_head
    if _metric_head is None:
        head = CardMetricHead(input_dim=512, hidden_dim=512, output_dim=256)
        head.load_state_dict(torch.load("models/metric-head/v1.../head.pt"))
        head.eval().to(EMBEDDING_DEVICE)
        _metric_head = head
    return _metric_head
```

In the search path, project before querying Qdrant:

```python
clip_vec   = encode(image)                          # 512-dim
metric_vec = get_metric_head()(clip_vec)            # 256-dim, L2-normalised
hits       = qdrant.search(query_vector=("metric", metric_vec))
```

### 7.3 Qdrant collection update (zero-downtime)

Add a new named vector field `metric` (256-dim) to the Qdrant collection alongside the existing `image` field. Backfill all points from S3 (load existing CLIP embeddings, project through head, write as new vector field). Once backfill completes and eval passes, switch the GPU worker to query `metric` instead of `image`. The old `image` vectors remain as a rollback path.

---

## Phase 8: Dataset Size Targets

| Dataset | Min useful | Good | Target |
|---------|-----------|------|--------|
| Unique identity keys | 5,000 | 50,000 | 200,000+ |
| Images per key (median) | 2 | 5 | 10+ |
| Total images | 10,000 | 250,000 | 2M+ |
| Hard negative pairs | 50,000 | 500,000 | 5M+ |

Given the OpenSearch corpus (millions of eBay sold listings), the target is achievable without any manual labelling — the seller-provided itemSpecifics are the label source.

---

## Implementation Timeline

| Week | Work |
|------|------|
| 1 | Data extraction script — scroll OS, filter, build identity keys, write parquet |
| 1 | Embedding extraction — extend `batch_job.py` to write `identity_key` column |
| 2 | DataLoader — parquet → batched SupCon-ready tensors with hard negative sampling |
| 2 | `CardMetricHead` + SupCon loss implementation |
| 3 | Training run on D3 (combined), eval against raw CLIP baseline |
| 3 | Tune temperature, hard negative ratio, output dim |
| 4 | Qdrant backfill of metric vectors, A/B test in GPU worker |

The first real signal on whether this is working comes at the end of Week 3 when you compare Recall@10 for the metric head vs raw CLIP on the held-out test set. If the gap is ≥ 5pp on hard negatives, the integration is worth doing.

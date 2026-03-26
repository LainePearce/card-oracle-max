# Card Search Platform — Vector Search Migration
## Claude Code Development Context

> **Document Reference:** DEV-VSM-2026-001  
> **Related Documents:** SAD-VSA-2026-001 (System Analysis & Design), EXP-VSC-2026-001 (Experimental Analysis)  
> **Last Updated:** March 2026  
> **Revision:** 1.5 — S3 vector store defined as primary durable store; naming convention and repopulation pipeline documented  
> **Status:** Active Development

---

## 1. Project Overview

This codebase implements the migration of the trading card search platform's vector search infrastructure from **Amazon OpenSearch Service KNN** to **Qdrant ANN**, as specified in SAD-VSA-2026-001. The project follows a **two-stage development process**. Stage 2 only begins if Stage 1 produces a successful experimental outcome.

---

### Stage 1 — Qdrant Comparative Analysis ← **CURRENT STAGE**

**Objective:** Populate a Qdrant database from the extant read-only OpenSearch cluster and run a direct comparative analysis of vector search capabilities between the two systems.

**In scope:**
- Stand up Qdrant (local dev → production cluster)
- Scroll the extant OpenSearch cluster to read all documents and generate embeddings
- Load all vectors and filterable payload into Qdrant
- Execute the experiment defined in EXP-VSC-2026-001 (Recall@K, latency, cost) comparing OpenSearch KNN vs Qdrant ANN side-by-side
- Produce a statistically validated experimental report

**Out of scope in Stage 1:**
- Any new OpenSearch cluster (does not exist yet)
- Any writes to the extant OpenSearch cluster (strictly read-only throughout)
- Production application traffic routing to Qdrant
- itemSpecifics inference pipeline for non-eBay sources (deferred to Stage 2)

**Stage 1 success criteria:** All primary null hypotheses rejected (H₁–H₃) with large effect sizes, and H₄ equivalence confirmed at Recall@10 ≥ 0.90. See EXP-VSC-2026-001 for full criteria.

---

### Stage 2 — Production System (Conditional on Stage 1 Success)

**Objective:** Stand up a new OpenSearch cluster to support the Qdrant-based production system, populated from the extant cluster.

**In scope:**
- Design the new OpenSearch index structure and document schema (informed by extant mapping but simplified — no vectors)
- Provision the new right-sized OpenSearch cluster (3× r6g.2xlarge, no KNN)
- Build a migration pipeline: extant OpenSearch → new OpenSearch + Qdrant (dual population)
- Build the itemSpecifics inference pipeline for non-eBay marketplace indices
- Route production application traffic to the new Qdrant + new OpenSearch stack
- Decommission vector fields from extant OpenSearch (or retire indices entirely)

**Stage 2 is not planned or built until Stage 1 experimental results are reviewed and approved.**

---

## 2. Repository Structure

> The repository is structured to reflect the two-stage process. Stage 1 files are active. Stage 2 directories are placeholders — **do not build Stage 2 content until Stage 1 is approved.**

```
card-search-infra/
├── CLAUDE.md                              # This file — read first
│
├── ── STAGE 1 (active) ──────────────────────────────────────────
│
├── docker/
│   ├── docker-compose.dev.yml             # Local single-node Qdrant (Stage 1 dev)
│   ├── docker-compose.test.yml            # Local 3-node Qdrant cluster (HA testing)
│   └── qdrant-config.yaml                 # Qdrant node configuration
│
├── infra/
│   ├── terraform/
│   │   ├── qdrant-cluster/                # EC2 r6gd.8xlarge × 3, NLB, SGs  [Stage 1]
│   │   └── batch-embedding/               # AWS Batch GPU job definitions      [Stage 1]
│   └── scripts/
│       ├── mount-nvme.sh                  # NVMe instance store mount for r6gd
│       └── bootstrap-qdrant.sh            # Docker install + Qdrant startup
│
├── src/
│   ├── schema/
│   │   └── collection.py                  # Qdrant collection creation + config [Stage 1]
│   ├── embeddings/
│   │   ├── image_encoder.py               # CLIP ViT-L/14 image embedding       [Stage 1]
│   │   ├── text_encoder.py                # MiniLM-L6-v2 specifics embedding    [Stage 1]
│   │   ├── projection.py                  # Image→specifics cross-modal projection [Stage 1]
│   │   ├── vector_store.py                # S3 vector persistence layer          [Stage 1]
│   │   └── batch_job.py                   # AWS Batch worker entry point         [Stage 1]
│   ├── ingestion/
│   │   ├── opensearch_reader.py           # Scroll extant OS (read-only)        [Stage 1]
│   │   ├── qdrant_writer.py               # Bulk upsert to Qdrant               [Stage 1]
│   │   └── pipeline.py                    # Orchestration: OS scroll → embed → Qdrant [Stage 1]
│   └── search/
│       ├── qdrant_search.py               # Qdrant query patterns               [Stage 1]
│       ├── opensearch_search.py           # OpenSearch KNN queries (control)    [Stage 1]
│       └── fusion.py                      # RRF score fusion + catalogue boost  [Stage 1]
│
├── experiment/                            # Full experiment suite                [Stage 1]
│   ├── README.md
│   ├── dataset/
│   │   ├── build_query_set.py             # Stratified query set construction
│   │   ├── build_ground_truth.py          # Expert annotation tooling
│   │   └── brute_force_reference.py       # Exact KNN reference set generation
│   ├── runner/
│   │   ├── run_experiment.py              # Main experiment orchestrator
│   │   ├── latency_harness.py             # Concurrent latency measurement
│   │   └── recall_evaluator.py            # Recall@K, nDCG@K computation
│   ├── analysis/
│   │   ├── statistical_tests.py           # Wilcoxon SR, TOST, Hochberg correction
│   │   ├── cost_model.py                  # Infrastructure cost projection
│   │   └── report_generator.py            # Results → structured output
│   └── results/                           # Experiment output artefacts (git-ignored)
│
├── ── STAGE 2 (placeholder — do not build until Stage 1 approved) ─
│
│   stage2/
│   ├── README.md                          # Stage 2 planning notes
│   ├── infra/
│   │   └── terraform/
│   │       └── opensearch-new/            # New right-sized OS cluster (no KNN)
│   ├── src/
│   │   ├── schema/
│   │   │   └── new_os_index_mapping.py    # New OS index design (derived from extant)
│   │   ├── embeddings/
│   │   │   └── specifics_inferrer.py      # itemSpecifics inference for non-eBay indices
│   │   ├── ingestion/
│   │   │   ├── migration_pipeline.py      # Extant OS → new OS + Qdrant dual-population
│   │   │   └── new_os_writer.py           # Writes to new OS only (never extant)
│   │   └── api/
│   │       └── search_handler.py          # Production query entry point (new stack)
│   └── docs/
│       ├── new_os_index_design.md         # New index structure rationale
│       └── migration_runbook.md           # Step-by-step production cutover
│
├── ── SHARED ────────────────────────────────────────────────────
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── requirements.txt
├── requirements-dev.txt
└── .env.example
```

---

## 3. System Architecture

### 3.1 Architecture by Stage

#### Stage 1 Architecture (Current — Experiment Only)

In Stage 1 there is no application traffic and no new OpenSearch cluster. The architecture is purely the experiment harness: the extant OpenSearch cluster (control) and Qdrant (treatment) are queried in parallel by the experiment runner.

```
┌──────────────────────────────────────────────────────────────┐
│                    EXPERIMENT RUNNER                          │
│              experiment/runner/run_experiment.py             │
└──────────┬───────────────────────────────────────┬───────────┘
           │ same query set                         │ same query set
           ▼                                        ▼
┌─────────────────────┐                 ┌────────────────────────┐
│  EXTANT OPENSEARCH  │                 │   QDRANT CLUSTER       │
│  (read-only)        │                 │   3× r6gd.8xlarge      │
│  us-west-1 AOS      │                 │   NLB → port 6334      │
│                     │                 │                        │
│  KNN search on      │                 │  Named vectors:        │
│  imageVector field  │                 │  "image"  512-dim      │
│  (control system)   │                 │  "specifics" 384-dim   │
│                     │                 │                        │
│  Also: data source  │ ─── scroll ──▶  │  INT8 quantization     │
│  for Qdrant backfill│                 │  On-disk HNSW          │
└─────────────────────┘                 │  Rescore top-200       │
                                        │  (treatment system)    │
                                        └────────────────────────┘
           │                                        │
           └──────────────┬─────────────────────────┘
                          ▼
              Recall@K / nDCG / latency / cost
              statistical comparison → report
```

#### Stage 2 Architecture (Planned — Not Built Yet)

Stage 2 replaces the experiment harness with a production query path and introduces a new right-sized OpenSearch cluster for document storage.

```
┌──────────────────────────────────────────────────────────────┐
│                     CLIENT APPLICATION                        │
└──────────────────────────┬───────────────────────────────────┘
                           │ query
                           ▼
                 src/api/search_handler.py
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
   ┌─────────────────────┐   ┌─────────────────────────┐
   │  QDRANT CLUSTER     │   │  NEW OPENSEARCH          │
   │  (same as Stage 1)  │   │  3× r6g.2xlarge          │
   │  vector search only │   │  Documents only — no KNN │
   └─────────────────────┘   └─────────────────────────┘
              │                         ▲
              │ top-K ids               │ mget enrichment
              └────────────────────────►│
                                        │
                                  merged results
```

### 3.2 Document Types

| Type | image vector | specifics vector | sold data | Notes |
|------|:---:|:---:|:---:|-------|
| **Type 1** Sold Listing | ✓ | ✓ | ✓ | Primary search target |
| **Type 2** Catalogue + Image | ✓ | ✓ | ✗ | Boost +0.12 on strong match |
| **Type 3** Catalogue Only | ✗ | ✓ | ✗ | Boost +0.15 if sim ≥ 0.90 |

### 3.3 Query Pipeline (Two-Hop)

```
Step 1  Encode query          ~20-50ms  GPU  CLIP ViT-L/14 or MiniLM
Step 2  Qdrant ANN search     ~10-25ms       Three-arm prefetch + RRF
Step 3  OpenSearch mget       ~5-15ms        Enrich top-K with full docs
Step 4  Merge + boost         <1ms           Score adjustment + sort
────────────────────────────────────────────
Total                         ~35-91ms  p99 target < 100ms
```

---

## 4. Qdrant Configuration

### 4.1 Collection Schema

**Collection name:** `cards`

```python
# src/schema/collection.py — canonical configuration
COLLECTION_CONFIG = {
    "collection_name": "cards",
    "vectors_config": {
        "image": VectorParams(
            size=512,
            distance=Distance.COSINE,
            on_disk=True,
            hnsw_config=HnswConfigDiff(
                m=16,
                ef_construct=200,
                on_disk=True,
            ),
        ),
        "specifics": VectorParams(
            size=384,
            distance=Distance.COSINE,
            on_disk=True,
            hnsw_config=HnswConfigDiff(
                m=16,
                ef_construct=200,
                on_disk=True,
            ),
        ),
    },
    "quantization_config": QuantizationConfig(
        scalar=ScalarQuantizationConfig(
            type=ScalarType.INT8,
            quantile=0.99,
            always_ram=True,   # quantized index hot in RAM; full vectors on disk
        )
    ),
    "optimizers_config": OptimizersConfigDiff(
        indexing_threshold=50_000,
        memmap_threshold=50_000,
    ),
    "on_disk_payload": True,
    "shard_number": 6,             # 2 per node in 3-node cluster
    "replication_factor": 2,
    "write_consistency_factor": 1,
}
```

### 4.2 Payload Schema

Every point stored in Qdrant must carry this payload. The fields are sourced directly from the OpenSearch index mapping (see Section 18). Keep payload **minimal** — the full source document remains in OpenSearch (read-only); payload exists only for Qdrant-side filtering and boosting. OpenSearch `mget` is used to enrich search results with full document data after Qdrant returns candidate IDs.

```python
# src/schema/collection.py — payload field definitions
# All fields sourced from itemSpecifics in the OpenSearch mapping
# plus system-level control fields

# ── Core identity fields (filterable, always present) ─────────────
PAYLOAD_REQUIRED = {
    "os_id":        str,   # OpenSearch document _id — primary join key between systems
    "item_id":      str,   # itemId from OpenSearch (eBay listing ID or catalogue ref)
    "type":         str,   # "sold" | "catalogue"
    "has_image":    bool,  # True if image vector is populated
    "source":       str,   # source system keyword (e.g. "ebay_uk", "ebay_us")
    "global_id":    str,   # globalId from OpenSearch (eBay site: EBAY-GB, EBAY-US, etc.)
    "active":       bool,  # soft-delete flag
}

# ── itemSpecifics fields (filterable — used for card ID and near-dup filtering) ─
# These map 1-to-1 from itemSpecifics in the OpenSearch index mapping.
# All keyword fields; normalised to lowercase at index time by lowercase_normalizer.
PAYLOAD_ITEM_SPECIFICS = {
    "brand":         str,  # card manufacturer / publisher
    "player":        str,  # athlete or character name
    "genre":         str,  # sport or card game genre (e.g. "football", "pokemon")
    "country":       str,  # country of origin / print run
    "set":           str,  # card set name — primary filter for near-dup suppression
    "card_number":   str,  # cardNumber within the set (e.g. "4/102")
    "subset":        str,  # subset or insert set name
    "parallel":      str,  # parallel variant (e.g. "holo", "reverse holo", "gold")
    "serial_number": str,  # serial number for numbered parallels (e.g. "42/99")
    "year":          int,  # release year
    "graded":        bool, # whether the card is graded
    "grader":        str,  # grading company (e.g. "psa", "bgs", "cgc")
    "grade":         str,  # grade value (e.g. "9", "9.5", "10")
    "type":          str,  # card type (e.g. "rookie", "base", "insert")
    "autographed":   bool, # whether the card is autographed
    "team":          str,  # team name
}

# ── Pricing / sale fields (used for result enrichment context) ────
# These are NOT used for Qdrant filtering — OpenSearch handles price range queries.
# Stored here only so fusion.py can make boost decisions without an OS round-trip.
PAYLOAD_SALE_CONTEXT = {
    "sale_type":     str,  # saleType: "FixedPrice" | "Auction"
    "current_price": float,# currentPrice snapshot at index time
    "currency":      str,  # currentPriceCurrency
}

# ── System fields (populated by ingestion pipeline) ───────────────
PAYLOAD_SYSTEM = {
    "image_is_pseudo":           bool,        # True if vector averaged from sold cards
    "pseudo_vector_sample_size": int,         # n sold cards used for pseudo vector
    "indexed_at":                str,         # ISO8601 timestamp of Qdrant write

    # ── Provenance — itemSpecifics data quality ───────────────────
    # ALWAYS populated. Consumers must check this before relying on
    # specifics fields for card identification or boosting decisions.
    "specifics_source":     str,              # "ebay" | "inferred" | "none"
    "specifics_confidence": float | None,     # 0.0–1.0 for "inferred"; None for "ebay"
    "specifics_ref_ids":    list | None,      # Qdrant IDs of eBay reference listings used
    "specifics_model":      str | None,       # model version string for "inferred"
}
```

> **Important — normalisation:** The OpenSearch mapping uses `lowercase_normalizer` on all `itemSpecifics` keyword fields. All values written to Qdrant payload **must be lowercased** before storage so that Qdrant `MatchValue` filters behave identically to OpenSearch term queries. Apply `value.lower().strip()` to every string in `PAYLOAD_ITEM_SPECIFICS` at write time (see `src/ingestion/qdrant_writer.py → normalise_payload()`).

> **What stays in OpenSearch only:** `title`, `galleryURL`, `itemURL`, `bidCount`, `salePrice`, `shippingServiceCost`, `endTime`, `BestOfferPrice` — these are for display and aggregation, never needed for Qdrant-side filtering.

### 4.3 Search Parameters

```python
# Standard search — image similarity
SEARCH_PARAMS_STANDARD = SearchParams(
    hnsw_ef=128,
    exact=False,
    quantization=QuantizationSearchParams(rescore=True),
)

# Hard query search — higher ef for difficult near-duplicate cases
# Trigger: filter reduces candidate pool below 500k OR specifics_sim > 0.95
SEARCH_PARAMS_HARD = SearchParams(
    hnsw_ef=256,
    exact=False,
    quantization=QuantizationSearchParams(rescore=True),
)

# Exact fallback — for card ID with very high confidence specifics match
# Trigger: specifics_sim > 0.95 AND filtered set < 10k items
SEARCH_PARAMS_EXACT = SearchParams(exact=True)
```

### 4.4 Score Boost Policy

```python
# src/search/fusion.py — catalogue card boost rules

# Base boost values — applied when specifics_source == "ebay"
BOOST_RULES = {
    "sold":                   0.00,   # baseline — no adjustment
    "catalogue_image":        0.12,   # has image vector; competed fairly in image search
    "catalogue_no_img_high":  0.15,   # specifics_sim >= 0.90
    "catalogue_no_img_mid":   0.05,   # specifics_sim 0.80–0.89
    "catalogue_no_img_low":  -0.10,   # specifics_sim < 0.80 — push down
}

# Multiplier applied to boost when specifics were inferred (not eBay-provided)
# Inferred specifics are lower-confidence — reduce their influence proportionally
INFERRED_SPECIFICS_BOOST_MULTIPLIER = 0.5

def apply_boost(score: float, point_payload: dict, specifics_sim: float) -> float:
    card_type  = point_payload.get("type", "sold")
    has_image  = point_payload.get("has_image", False)
    src        = point_payload.get("specifics_source", "ebay")  # "ebay"|"inferred"|"none"
    multiplier = INFERRED_SPECIFICS_BOOST_MULTIPLIER if src == "inferred" else 1.0

    if card_type == "sold":
        return score   # no adjustment regardless of specifics source

    if card_type == "catalogue" and has_image:
        boost = BOOST_RULES["catalogue_image"]
    elif card_type == "catalogue" and not has_image:
        if specifics_sim >= 0.90:
            boost = BOOST_RULES["catalogue_no_img_high"]
        elif specifics_sim >= 0.80:
            boost = BOOST_RULES["catalogue_no_img_mid"]
        else:
            boost = BOOST_RULES["catalogue_no_img_low"]
    else:
        boost = 0.0

    return score + (boost * multiplier)
```

---

## 5. Embedding Models

### 5.1 Image Embeddings — CLIP ViT-L/14

```python
# src/embeddings/image_encoder.py
MODEL_NAME = "ViT-L/14"          # OpenCLIP
EMBEDDING_DIM = 512
BATCH_SIZE = 256                  # tuned for g5.xlarge (24GB VRAM)
USE_FP16 = True                   # torch.cuda.amp.autocast()
NORMALISE = True                  # L2 normalise all output vectors

# Preprocessing: standard CLIP transforms (resize 224, centre crop, normalise)
# CRITICAL: use identical preprocessing at index time and query time
```

### 5.2 Specifics Embeddings — MiniLM-L6-v2

```python
# src/embeddings/text_encoder.py
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
BATCH_SIZE = 1024                 # CPU-friendly; can run in parallel with GPU image encoding

# Input format: flatten itemSpecifics from the OpenSearch mapping into a
# natural language sentence. Field names match the OpenSearch mapping exactly.
# Fields are ordered by discriminative power for card identification:
#   player/brand (who) → set (which set) → card_number (exact card) →
#   year → subset/parallel (variant) → grader/grade (condition)
#
# CRITICAL: apply the same lowercase + strip normalisation as
# OpenSearch's lowercase_normalizer before encoding.

def format_specifics(item_specifics: dict) -> str:
    """
    Converts an itemSpecifics dict (from OpenSearch source document) into
    a single natural language string for MiniLM encoding.

    Handles missing fields gracefully — absent fields contribute empty string.
    Handles noisy data: typos and abbreviations are absorbed by the semantic model.
    """
    parts = []

    # Primary identity
    if player := item_specifics.get("player", "").strip():
        parts.append(player)
    if brand := item_specifics.get("brand", "").strip():
        parts.append(brand)
    if genre := item_specifics.get("genre", "").strip():
        parts.append(genre)

    # Card location in set
    if set_name := item_specifics.get("set", "").strip():
        parts.append(set_name)
    if card_number := item_specifics.get("cardNumber", "").strip():
        parts.append(f"number {card_number}")
    if year := item_specifics.get("year"):
        parts.append(str(year))

    # Variant information
    if subset := item_specifics.get("subset", "").strip():
        parts.append(subset)
    if parallel := item_specifics.get("parallel", "").strip():
        parts.append(parallel)
    if serial := item_specifics.get("serialNumber", "").strip():
        parts.append(f"serial {serial}")

    # Condition / grading
    if item_specifics.get("graded"):
        grader = item_specifics.get("grader", "").strip()
        grade  = item_specifics.get("grade", "").strip()
        if grader and grade:
            parts.append(f"{grader} {grade}")
        elif grader:
            parts.append(grader)

    if item_specifics.get("autographed"):
        parts.append("autograph")

    # Geographic / team context
    if team := item_specifics.get("team", "").strip():
        parts.append(team)
    if country := item_specifics.get("country", "").strip():
        parts.append(country)

    return " ".join(p.lower() for p in parts if p)


# Also used when building the text from the listing title as fallback
# when itemSpecifics are sparse or missing:
def format_specifics_from_title(title: str) -> str:
    """Fallback: encode the raw listing title. Lower quality but better than empty."""
    return title.lower().strip()
```

### 5.3 Cross-Modal Projection (Image → Specifics Space)

```python
# src/embeddings/projection.py
# Small learned projection: 512-dim CLIP → 384-dim MiniLM
# Trained on sold cards (both vectors known) using cosine similarity loss
# Required for querying Type 3 catalogue cards (no image) via image query

PROJECTION_MODEL_PATH = "models/image_to_specifics_projection.pt"

class ImageToSpecificsProjection(nn.Module):
    def __init__(self, image_dim=512, specifics_dim=384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(image_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, specifics_dim),
            nn.LayerNorm(specifics_dim),
        )

    def forward(self, x):
        projected = self.net(x)
        return projected / projected.norm(dim=-1, keepdim=True)  # L2 normalise
```

---

## 6. Search Query Patterns

### 6.1 Image Similarity Search (Primary Use Case)

```python
# src/search/qdrant_search.py
async def image_search(
    client: QdrantClient,
    image_embedding: np.ndarray,
    projected_specifics: np.ndarray,
    filters: dict | None = None,
    top_k: int = 20,
    is_hard_query: bool = False,
) -> list[ScoredPoint]:
    """
    Three-arm prefetch with RRF fusion:
      Arm 1: image ANN — all cards with has_image=True
      Arm 2: specifics ANN — all cards (sold + catalogue with + without image)
      Arm 3: specifics ANN scoped to has_image=False catalogue only
             (gives no-image cards a separate pool so they aren't drowned out)
    """
    params = SEARCH_PARAMS_HARD if is_hard_query else SEARCH_PARAMS_STANDARD

    results = client.query_points(
        collection_name="cards",
        prefetch=[
            Prefetch(
                query=image_embedding.tolist(),
                using="image",
                limit=150,
                filter=build_filter({"has_image": True, **(filters or {})}),
                params=params,
            ),
            Prefetch(
                query=projected_specifics.tolist(),
                using="specifics",
                limit=150,
                filter=build_filter(filters) if filters else None,
                params=params,
            ),
            Prefetch(
                query=projected_specifics.tolist(),
                using="specifics",
                limit=100,
                filter=build_filter({"has_image": False, **(filters or {})}),
                params=params,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k * 3,   # over-fetch before boost re-ranking
        with_payload=True,
    )
    return results.points
```

### 6.2 Card Identification Search

```python
async def card_identification_search(
    client: QdrantClient,
    specifics_embedding: np.ndarray,
    image_embedding: np.ndarray | None = None,
    filters: dict | None = None,
    top_k: int = 20,
) -> list[ScoredPoint]:
    """
    When both image and specifics are available use RRF.
    When only specifics available (Type 3 query) use specifics arm only.
    """
    prefetch_arms = [
        Prefetch(
            query=specifics_embedding.tolist(),
            using="specifics",
            limit=150,
            filter=build_filter(filters) if filters else None,
        ),
    ]
    if image_embedding is not None:
        prefetch_arms.append(
            Prefetch(
                query=image_embedding.tolist(),
                using="image",
                limit=150,
                filter=build_filter({"has_image": True, **(filters or {})}),
            )
        )

    results = client.query_points(
        collection_name="cards",
        prefetch=prefetch_arms,
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k * 3,
        with_payload=True,
    )
    return results.points
```

### 6.3 OpenSearch Equivalent (Control System)

```python
# src/search/opensearch_search.py — used in experiment only (control condition)
#
# Queries the live AOS cluster at:
#   https://search-es130point-vector-h3eaau7mwcpyhgynkthfcwzeje.aos.us-west-1.on.aws
#
# imageVector is the field name in the OpenSearch mapping (camelCase).
# Queries target the per-date index pattern — use "*" for full corpus search
# or a specific YYYY-MM-DD index to match the Qdrant experiment subset.
#
# EXPERIMENT NOTE: post_filter is applied AFTER KNN candidate selection in
# OpenSearch — this degrades recall when filters are selective. Do not
# pre-filter in the OpenSearch control query; measure the degradation as-is.
# This is a documented weakness of OpenSearch KNN (see SAD-VSA-2026-001 §2.5).

def opensearch_knn_search(
    client: OpenSearch,
    image_embedding: np.ndarray,
    index_pattern: str = "*",   # default: search all date indices
    filters: dict | None = None,
    top_k: int = 20,
) -> list[dict]:
    """
    Execute a KNN search against the live OpenSearch cluster.
    Returns raw hit dicts for the experiment runner to evaluate.
    """
    query: dict = {
        "size": top_k,
        "query": {
            "knn": {
                "imageVector": {          # field name from OS mapping (camelCase)
                    "vector": image_embedding.tolist(),
                    "k": top_k,
                }
            }
        },
        # Retrieve only the fields needed for experiment evaluation and payload join.
        # imageVector / textVector are excluded from _source — do not request them.
        "_source": [
            "id", "itemId", "source", "globalId",
            "galleryURL", "saleType", "currentPrice",
            "itemSpecifics",
        ],
    }

    # post_filter: applied after KNN — intentionally causes recall degradation
    # when selective, which is what we are measuring in the experiment.
    if filters:
        query["post_filter"] = build_os_filter(filters)

    response = client.search(index=index_pattern, body=query)
    return response["hits"]["hits"]


def build_os_filter(criteria: dict) -> dict:
    """
    Build an OpenSearch bool filter from itemSpecifics criteria.
    Maps snake_case experiment keys back to the camelCase OS mapping field names.
    """
    field_map = {
        "set":         "itemSpecifics.set",
        "card_number": "itemSpecifics.cardNumber",
        "grader":      "itemSpecifics.grader",
        "grade":       "itemSpecifics.grade",
        "graded":      "itemSpecifics.graded",
        "genre":       "itemSpecifics.genre",
        "player":      "itemSpecifics.player",
        "brand":       "itemSpecifics.brand",
        "parallel":    "itemSpecifics.parallel",
        "year":        "itemSpecifics.year",
        "autographed": "itemSpecifics.autographed",
        "team":        "itemSpecifics.team",
        "source":      "source",
        "global_id":   "globalId",
    }
    must_clauses = []
    for key, value in criteria.items():
        os_field = field_map.get(key, key)
        if isinstance(value, bool):
            must_clauses.append({"term": {os_field: value}})
        elif isinstance(value, str):
            must_clauses.append({"term": {os_field: value.lower().strip()}})
        elif isinstance(value, (int, float)):
            must_clauses.append({"term": {os_field: value}})
    return {"bool": {"must": must_clauses}}
```

---

## 7. Ingestion Pipeline

### 7.1 ID Strategy

```python
# CRITICAL: IDs must be consistent between Qdrant and OpenSearch
# Qdrant requires uint64 or UUID
# OpenSearch uses the numeric `id` field (type: long) as the document identifier.
# Map this directly to Qdrant uint64 where possible; fall back to UUID5 for
# string-based IDs (e.g. catalogue items without a numeric eBay ID).

import uuid

NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

def os_id_to_qdrant_id(opensearch_id: str | int) -> str | int:
    """
    Prefer numeric: if the OpenSearch `id` field is a long integer,
    use it directly as a Qdrant uint64 — no conversion needed.
    Otherwise derive a deterministic UUID5 from the string _id.
    """
    if isinstance(opensearch_id, int):
        return opensearch_id          # uint64 — direct passthrough
    try:
        return int(opensearch_id)     # numeric string — cast
    except ValueError:
        return str(uuid.uuid5(NAMESPACE, opensearch_id))  # UUID fallback

def qdrant_id_to_os_id(qdrant_point: ScoredPoint) -> str:
    """Reverse lookup — always use the os_id stored in payload."""
    return qdrant_point.payload["os_id"]
```

### 7.2 OpenSearch Scroll Reader

```python
# src/ingestion/opensearch_reader.py
#
# Connects to the live AOS cluster:
#   https://search-es130point-vector-h3eaau7mwcpyhgynkthfcwzeje.aos.us-west-1.on.aws
#
# The index uses per-date indices named YYYY-MM-DD. The reader scrolls
# across all indices matching the supplied date pattern.
#
# imageVector and textVector are excluded from _source — they exist in
# the Lucene index for KNN traversal but CANNOT be retrieved via scroll.
# Embeddings are re-generated from galleryURL (image) and itemSpecifics (text).

from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import boto3, os
from datetime import date, timedelta
from typing import Generator

SCROLL_PAGE_SIZE  = 1_000   # docs per scroll page
SCROLL_TIMEOUT    = "5m"    # keep scroll cursor alive

# Source fields to retrieve — vectors excluded (not in _source)
SCROLL_SOURCE_FIELDS = [
    "id", "itemId", "source", "globalId",
    "title", "galleryURL",
    "saleType", "currentPrice", "currentPriceCurrency",
    "endTime",
    "itemSpecifics",   # nested object with all card identity fields
]

def get_opensearch_client() -> OpenSearch:
    """Connect to the live AOS cluster using env-configured auth."""
    host = os.environ["OPENSEARCH_HOST"]
    use_iam = os.environ.get("OPENSEARCH_USE_IAM", "false").lower() == "true"

    if use_iam:
        credentials = boto3.Session().get_credentials()
        auth = AWS4Auth(
            credentials.access_key,
            credentials.secret_key,
            os.environ.get("AWS_REGION", "us-west-1"),
            "es",
            session_token=credentials.token,
        )
    else:
        auth = (os.environ["OPENSEARCH_USER"], os.environ["OPENSEARCH_PASSWORD"])

    return OpenSearch(
        hosts=[{"host": host, "port": int(os.environ.get("OPENSEARCH_PORT", 443))}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=60,
    )

def date_index_pattern(start: date, end: date) -> str:
    """
    Build a comma-separated list of YYYY-MM-DD index names for the date range.
    Avoids wildcard patterns that could accidentally match unintended indices.
    Example: date_index_pattern(date(2024,1,1), date(2024,1,3)) -> "2024-01-01,2024-01-02,2024-01-03"
    """
    indices = []
    current = start
    while current <= end:
        indices.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return ",".join(indices)

def scroll_index(
    client: OpenSearch,
    index_pattern: str,
) -> Generator[dict, None, None]:
    """
    Scroll through all documents in indices matching index_pattern.

    index_pattern examples:
      "2024-01-15"              — single date index
      "2024-01-01,2024-01-02"  — explicit list (from date_index_pattern())
      "2024-*"                 — wildcard (use with caution)
      "*"                      — all date indices (full backfill)

    Yields raw OpenSearch hit dicts: {"_id": ..., "_source": {...}}
    """
    resp = client.search(
        index=index_pattern,
        scroll=SCROLL_TIMEOUT,
        size=SCROLL_PAGE_SIZE,
        body={"query": {"match_all": {}}, "_source": SCROLL_SOURCE_FIELDS},
    )
    scroll_id = resp["_scroll_id"]
    hits = resp["hits"]["hits"]

    while hits:
        yield from hits
        resp = client.scroll(scroll_id=scroll_id, scroll=SCROLL_TIMEOUT)
        scroll_id = resp["_scroll_id"]
        hits = resp["hits"]["hits"]

    client.clear_scroll(scroll_id=scroll_id)
```

### 7.3 Payload Extraction

```python
# src/ingestion/qdrant_writer.py

def extract_payload(os_hit: dict) -> dict:
    """
    Build Qdrant payload from an OpenSearch scroll hit.
    Field names follow the OpenSearch mapping exactly.
    All string fields normalised to lowercase (mirrors lowercase_normalizer).
    """
    src = os_hit["_source"]
    specs = src.get("itemSpecifics", {}) or {}

    def lower(val) -> str:
        return str(val).lower().strip() if val is not None else ""

    return {
        # ── Core identity ─────────────────────────────
        "os_id":        os_hit["_id"],
        "item_id":      str(src.get("itemId", "")),
        "type":         "sold",       # set to "catalogue" for catalogue docs
        "has_image":    bool(src.get("galleryURL")),
        "source":       lower(src.get("source")),
        "global_id":    lower(src.get("globalId")),
        "active":       True,

        # ── itemSpecifics (all lowercased) ─────────────
        "brand":         lower(specs.get("brand")),
        "player":        lower(specs.get("player")),
        "genre":         lower(specs.get("genre")),
        "country":       lower(specs.get("country")),
        "set":           lower(specs.get("set")),
        "card_number":   lower(specs.get("cardNumber")),
        "subset":        lower(specs.get("subset")),
        "parallel":      lower(specs.get("parallel")),
        "serial_number": lower(specs.get("serialNumber")),
        "year":          int(specs["year"]) if specs.get("year") else None,
        "graded":        bool(specs.get("graded", False)),
        "grader":        lower(specs.get("grader")),
        "grade":         lower(specs.get("grade")),
        "card_type":     lower(specs.get("type")),   # renamed to avoid clash with doc type
        "autographed":   bool(specs.get("autographed", False)),
        "team":          lower(specs.get("team")),

        # ── Sale context ──────────────────────────────
        "sale_type":     lower(src.get("saleType")),
        "current_price": float(src["currentPrice"]) if src.get("currentPrice") else None,
        "currency":      lower(src.get("currentPriceCurrency")),

        # ── System ────────────────────────────────────
        "image_is_pseudo":           False,
        "pseudo_vector_sample_size": 0,
        "indexed_at":                datetime.utcnow().isoformat(),
    }
```

### 7.4 Batch Upsert

```python
# src/ingestion/qdrant_writer.py
UPSERT_BATCH_SIZE = 1_000    # points per upsert call
UPSERT_WAIT = False          # async — don't block per batch (use for backfill)
CHECKPOINT_EVERY = 10_000    # save progress checkpoint to S3 after N points

async def upsert_batch(client: QdrantClient, points: list[PointStruct]) -> None:
    client.upsert(
        collection_name="cards",
        points=points,
        wait=UPSERT_WAIT,
    )
```

### 7.5 Data Flow by Development Stage

> **The extant OpenSearch cluster is strictly read-only throughout both stages.**  
> No code in this repository writes, updates, deletes, or re-indexes any document in the extant cluster at any point.

#### Stage 1 Data Flow (Current)

```
Extant OpenSearch cluster  (read-only)
    │
    │  scroll all indices — read documents, itemSpecifics, galleryURL
    ▼
Embedding pipeline
    │  CLIP ViT-L/14    → image vector  (512-dim)
    │  MiniLM-L6-v2     → specifics vector (384-dim)
    │
    ├──────────────────────────────────────┐
    ▼                                      ▼
S3 Vector Store (primary durable store)   (written first — mandatory)
    │  vectors/{type}/{model}/{params}/
    │  {index_type}/{partition}/{shard}.parquet
    │
    ▼
Qdrant                     (queryable index — rebuilt from S3 if lost)
    │  named vectors: image + specifics
    │  payload: filterable itemSpecifics fields + provenance
    ▼
Experiment runner
    │  queries both Qdrant ANN and extant OpenSearch KNN
    │  computes Recall@K, nDCG, latency, cost
    ▼
Experimental report        (EXP-VSC-2026-001)
```

#### Stage 2 Data Flow (Planned — conditional on Stage 1 approval)

```
Extant OpenSearch cluster  (still read-only)
    │
    │  scroll all indices
    │
    ├─────────────────────────────────┐
    ▼                                 ▼
New OpenSearch cluster           Qdrant cluster
(write: full documents,          (update: fill any gaps
 no vectors, no KNN)              from extant data)
    │                                 │
    └─────────────────┬───────────────┘
                      ▼
             Production application
             (queries Qdrant for vectors,
              new OpenSearch for documents)
```

Stage 2 pipeline code lives exclusively in `stage2/` and is **not activated by Stage 1 code paths**. The `DEVELOPMENT_STAGE` env var controls which paths are live.

---

## 8. Experiment Infrastructure

### 8.1 Hypotheses Being Tested

| ID | Null Hypothesis | Test | α | Primary Metric |
|----|----------------|------|---|----------------|
| H₁ | No recall difference between systems | Wilcoxon SR (one-tailed) | 0.05 | Recall@10 |
| H₂ | No latency difference | Wilcoxon SR (one-tailed) | 0.05 | p99 latency (ms) |
| H₃ | No cost difference >40% | Threshold test | 0.05 | Cost per 1M queries |
| H₄ | Qdrant exact recall < 0.90 | TOST equivalence | 0.05 | Exact Recall@10 |
| S₁ | No near-duplicate reduction | Wilcoxon SR | 0.05 | Dup rate in top-10 |
| S₂ | RRF no better than single-vec | Wilcoxon SR | 0.05 | Card ID Recall@10 |
| S₃ | Catalogue boost has no effect | Wilcoxon SR | 0.05 | Catalogue in top-10 |

**Multiple comparison correction:** Hochberg step-up procedure applied across all 7 tests.  
**Minimum required effect size:** Cohen's d ≥ 0.20 (small effect).  
**Required power:** 1-β ≥ 0.90.

### 8.2 Query Set Specification

```python
# experiment/dataset/build_query_set.py
# Stratification fields map directly to itemSpecifics in the OpenSearch mapping.

QUERY_SET_SPEC = {
    "image_search": {
        "total": 10_000,
        "stratification": {
            # genre (from itemSpecifics.genre — keyword, lowercase_normalizer)
            "genre": {
                "pokemon":  0.35,
                "football": 0.25,
                "basketball": 0.20,
                "baseball": 0.10,
                "other":    0.10,
            },
            # graded (from itemSpecifics.graded — boolean)
            # grader (from itemSpecifics.grader — keyword)
            "condition": {
                "raw":  0.40,   # graded=false
                "psa":  0.30,   # grader="psa"
                "bgs":  0.20,   # grader="bgs"
                "cgc":  0.10,   # grader="cgc"
            },
            # difficulty = number of near-duplicate cards in corpus
            # sharing the same (set, cardNumber) but different (parallel, grade)
            "difficulty": {"easy": 0.33, "medium": 0.33, "hard": 0.34},
        },
        "difficulty_thresholds": {
            "easy":   "<5 items sharing (set, cardNumber)",
            "medium": "5-20 items sharing (set, cardNumber)",
            "hard":   ">20 items sharing (set, cardNumber)",  # e.g. Charizard Base Set
        },
    },
    "card_id": {
        "total": 5_000,
        # Noise injection mirrors real-world data quality issues in itemSpecifics
        "noise_injection": {
            "typo_rate":           0.15,   # e.g. "charizrd" instead of "charizard"
            "missing_field_rate":  0.20,   # one or more of set/cardNumber/grade absent
            "abbreviation_rate":   0.10,   # e.g. "psa10" instead of grader="psa" grade="10"
            "wrong_case_rate":     0.05,   # mixed case before normalisation (model robustness test)
        },
        # Fields used as ground-truth identity for card ID task:
        # A match is "exact" if (set + cardNumber) match AND (grader + grade) match.
        # A match is "correct" if (set + cardNumber) match regardless of condition.
        "identity_fields": ["set", "cardNumber", "grader", "grade", "parallel"],
    },
}
```

### 8.3 Relevance Annotation Schema

```python
# Ground truth relevance scores for nDCG computation
RELEVANCE_SCHEMA = {
    0: "Irrelevant",        # different card entirely
    1: "Related",           # same card type / set, wrong specific card
    2: "Correct match",     # correct card identity, different condition/grade
    3: "Exact match",       # correct card identity AND condition/grade
}

# Inter-annotator agreement threshold
MIN_KAPPA = 0.70    # Cohen's Kappa — required before judgements accepted
# Disagreements resolved by majority vote
# Items with 3-way disagreement excluded from evaluation set
```

### 8.4 Metrics Implementation

```python
# experiment/runner/recall_evaluator.py

def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Proportion of relevant items found in top-K retrieved."""
    top_k = retrieved[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant) if relevant else 0.0

def ndcg_at_k(retrieved: list[str], relevance_scores: dict[str, int], k: int) -> float:
    """Normalised Discounted Cumulative Gain at rank K."""
    import math
    def dcg(ranking):
        return sum(
            relevance_scores.get(doc, 0) / math.log2(i + 2)
            for i, doc in enumerate(ranking[:k])
        )
    actual_dcg = dcg(retrieved)
    ideal_ranking = sorted(relevance_scores, key=relevance_scores.get, reverse=True)
    ideal_dcg = dcg(ideal_ranking)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0

def exact_recall_at_k(
    ann_results: list[str],
    exact_results: list[str],
    k: int,
) -> float:
    """Fraction of ANN top-K that appear in exact nearest-neighbour top-K."""
    ann_set = set(ann_results[:k])
    exact_set = set(exact_results[:k])
    return len(ann_set & exact_set) / k
```

### 8.5 Statistical Tests

```python
# experiment/analysis/statistical_tests.py
from scipy import stats
import numpy as np

def wilcoxon_one_tailed(
    system_a_scores: np.ndarray,
    system_b_scores: np.ndarray,
) -> tuple[float, float, float]:
    """
    One-tailed Wilcoxon signed-rank test: H₁ is System B > System A.
    Returns: (statistic, p_value_one_tailed, effect_size_r)
    """
    differences = system_b_scores - system_a_scores
    stat, p_two_tailed = stats.wilcoxon(differences, alternative="greater")
    n = len(differences)
    z = stats.norm.ppf(1 - p_two_tailed / 2)     # approximate Z from W
    effect_r = abs(z) / np.sqrt(n)
    return stat, p_two_tailed, effect_r

def hochberg_correction(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """
    Hochberg step-up procedure for multiple comparisons.
    Returns list of booleans: True = reject H₀.
    """
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1], reverse=True)
    reject = [False] * m
    for j, (orig_idx, p) in enumerate(indexed):
        threshold = alpha / (j + 1)
        if p <= threshold:
            # reject this and all remaining (lower p-values)
            for _, (idx, _) in enumerate(indexed[j:]):
                reject[idx] = True
            break
    return reject

def tost_equivalence(
    scores: np.ndarray,
    lower_bound: float = 0.90,
    upper_bound: float = 1.10,
    alpha: float = 0.05,
) -> tuple[bool, tuple[float, float]]:
    """
    Two One-Sided Tests for equivalence.
    Returns: (is_equivalent, confidence_interval_90pct)
    """
    mean = np.mean(scores)
    se = stats.sem(scores)
    ci_low, ci_high = stats.t.interval(0.90, len(scores)-1, loc=mean, scale=se)
    is_equivalent = ci_low >= lower_bound and ci_high <= upper_bound
    return is_equivalent, (ci_low, ci_high)
```

### 8.6 Latency Harness

```python
# experiment/runner/latency_harness.py
LATENCY_CONFIG = {
    "warmup_queries":     1_000,   # execute before measurement begins
    "measurement_queries": 15_000,
    "concurrent_threads":  50,
    "repetitions":         5,       # full cycle repeated 5 times
    "slo_p99_ms":          100,     # service-level objective
}

# Percentiles to record
PERCENTILES = [50, 95, 99, 100]    # p50, p95, p99, max

# Clock: time.perf_counter_ns() — nanosecond precision, convert to ms
# Measure from: query vector submission to result set received
# Exclude: embedding encoding time (measured separately)
```

---

## 9. Infrastructure Specifications

> Infrastructure is staged. Only Stage 1 components are provisioned now. Stage 2 components are planned but **not built until Stage 1 experimental results are approved**.

### 9.1 Stage 1 Infrastructure

#### 9.1.1 Extant OpenSearch Cluster (Read-Only Data Source)

| Parameter | Value |
|-----------|-------|
| Host | `search-es130point-vector-h3eaau7mwcpyhgynkthfcwzeje.aos.us-west-1.on.aws` |
| Region | `us-west-1` |
| Access | **Read-only** — scroll and KNN query only |
| Role in Stage 1 | Control system (KNN baseline) + data source for Qdrant backfill |
| Managed by | Upstream pipeline — outside scope of this codebase |

#### 9.1.2 Qdrant Cluster (Stage 1 — Vector Store and Treatment System)

| Parameter | Value |
|-----------|-------|
| Instance type | `r6gd.8xlarge` (256GB RAM, 1.9TB NVMe, 32 vCPU) |
| Node count | 3 |
| Shards | 6 (2 per node) |
| Replication factor | 2 |
| Storage | Instance NVMe (`/dev/nvme1n1` → `/mnt/qdrant-storage`) |
| Network | AWS NLB → port 6334 (gRPC) |
| API key | `QDRANT_API_KEY` env var |
| Role in Stage 1 | Treatment system — ANN search under experimental evaluation |
| Monthly cost (est.) | ~$4,267 on-demand |

### 9.2 Stage 2 Infrastructure (Planned — Not Yet Provisioned)

> Do not provision these resources until Stage 1 experimental results are reviewed and Stage 2 is formally approved.

#### 9.2.1 New OpenSearch Cluster (Stage 2 — Document Store Only, No KNN)

| Parameter | Value |
|-----------|-------|
| Data nodes | 3× `r6g.2xlarge` (64GB RAM, 8 vCPU) |
| Master nodes | 3× `r6g.xlarge` (32GB RAM, 4 vCPU) |
| EBS per data node | 600GB gp3 |
| Purpose | Document storage, BM25 text search, aggregations — **no KNN whatsoever** |
| Populated from | Extant OpenSearch cluster (read via scroll; write to new cluster only) |
| Monthly cost (est.) | ~$1,873 |
| Status | **Not provisioned — Stage 2 only** |

#### 9.2.2 Stage 2 Data Flow

```
Extant OpenSearch (read-only, us-west-1)
        │
        │  scroll all indices (eBay + marketplace)
        │
        ├──────────────────────────────────────────┐
        ▼                                          ▼
  New OpenSearch cluster                    Qdrant cluster
  (document store — no vectors)             (already populated in Stage 1)
  Write: full document minus vectors        Update: add any missing items
  Index design: see stage2/docs/            Qdrant remains primary vector store
```

The extant cluster is never modified. It is read in Stage 2 exactly as in Stage 1 — as a read-only data source.

### 9.3 Docker Compose — Local Development

```yaml
# docker/docker-compose.dev.yml
version: "3.8"
services:
  qdrant:
    image: qdrant/qdrant:v1.8.2
    restart: unless-stopped
    ports:
      - "6333:6333"   # HTTP REST
      - "6334:6334"   # gRPC
    volumes:
      - qdrant_storage:/qdrant/storage
    environment:
      - QDRANT__SERVICE__API_KEY=${QDRANT_API_KEY}
      - QDRANT__LOG_LEVEL=INFO
    ulimits:
      nofile: { soft: 65535, hard: 65535 }

volumes:
  qdrant_storage:
```

### 9.4 NVMe Mount Script (Production EC2)

```bash
# infra/scripts/mount-nvme.sh — run once on r6gd instance startup
set -euo pipefail
DEVICE="/dev/nvme1n1"
MOUNT="/mnt/qdrant-storage"

mkfs.ext4 -F "$DEVICE"
mkdir -p "$MOUNT"
mount "$DEVICE" "$MOUNT"
chown -R ec2-user:ec2-user "$MOUNT"
echo "$DEVICE $MOUNT ext4 defaults,nofail 0 2" >> /etc/fstab
echo "NVMe mounted at $MOUNT"
```

---

## 10. Environment Variables

```bash
# .env.example — copy to .env, never commit .env
# Actual credentials for the live OpenSearch cluster will be provided separately
# and must be added to your local .env — never committed to version control.

# ── Qdrant ────────────────────────────────────────────────────────
QDRANT_HOST=localhost                    # change to NLB DNS for production cluster
QDRANT_PORT=6334                         # gRPC port
QDRANT_API_KEY=your-secret-key
QDRANT_USE_GRPC=true
QDRANT_COLLECTION=cards

# ── OpenSearch — Live cluster (control system + backfill source) ──
# The existing production cluster that holds all sold listing data
# including imageVector and textVector (used for KNN, not in _source).
# This is the control system for the experiment and the data source
# for the Qdrant backfill. READ-ONLY — do not write to this cluster.
OPENSEARCH_HOST=search-es130point-vector-h3eaau7mwcpyhgynkthfcwzeje.aos.us-west-1.on.aws
OPENSEARCH_PORT=443
OPENSEARCH_USE_SSL=true
OPENSEARCH_VERIFY_CERTS=true

# Auth — choose ONE of the following two options:

# Option A: Basic auth (username/password)
# Fill these in with credentials provided separately.
OPENSEARCH_USE_IAM=false
OPENSEARCH_USER=your-username
OPENSEARCH_PASSWORD=your-password

# Option B: IAM-based auth (AWS4Auth via instance role or access keys)
# Set OPENSEARCH_USE_IAM=true and ensure AWS credentials are available
# via the standard AWS credential chain (instance profile, env vars, ~/.aws/credentials).
# OPENSEARCH_USE_IAM=true
# AWS_ACCESS_KEY_ID=your-key-id       # only needed if not using instance profile
# AWS_SECRET_ACCESS_KEY=your-secret   # only needed if not using instance profile

# ── OpenSearch — New document cluster (Stage 2 only — does not exist yet) ─
# These vars are commented out. Uncomment and populate only when Stage 2
# is approved and the new cluster is provisioned. Do not use in Stage 1.
# OPENSEARCH_DOCS_HOST=TBD
# OPENSEARCH_DOCS_PORT=443
# OPENSEARCH_DOCS_USE_SSL=true
# OPENSEARCH_DOCS_USER=TBD
# OPENSEARCH_DOCS_PASSWORD=TBD

# ── Development stage control ─────────────────────────────────────
# Controls which pipeline branches are active. Set to "1" for Stage 1.
# Changing to "2" enables Stage 2 code paths (migration pipeline, new OS writer).
# NEVER set to "2" until Stage 1 results are approved.
DEVELOPMENT_STAGE=1

# ── AWS ───────────────────────────────────────────────────────────
AWS_REGION=us-west-1                    # must match the live OS cluster region
AWS_BATCH_JOB_QUEUE=gpu-spot-queue
AWS_BATCH_JOB_DEFINITION=clip-embedding-job

# ── S3 Vector Store ───────────────────────────────────────────────
# Single bucket for all vectors. Key structure encodes all metadata.
# See Section 16 for full naming convention.
S3_VECTOR_BUCKET=your-vector-store-bucket

# Root prefix within the bucket (useful if bucket is shared with other data)
S3_VECTOR_PREFIX=vectors

# Separate prefix for progress checkpoints (written per-batch during backfill)
S3_CHECKPOINT_PREFIX=checkpoints

# ── Embedding models ──────────────────────────────────────────────
CLIP_MODEL=ViT-L/14
MINILM_MODEL=sentence-transformers/all-MiniLM-L6-v2
PROJECTION_MODEL_PATH=models/image_to_specifics_projection.pt
EMBEDDING_DEVICE=cuda                   # cuda | cpu
EMBEDDING_BATCH_SIZE_IMAGE=256
EMBEDDING_BATCH_SIZE_TEXT=1024

# ── Experiment ────────────────────────────────────────────────────
EXPERIMENT_ID=EXP-VSC-2026-001
EXPERIMENT_RESULTS_DIR=experiment/results
EXPERIMENT_QUERY_SET_PATH=experiment/dataset/query_set.parquet
EXPERIMENT_GROUND_TRUTH_PATH=experiment/dataset/ground_truth.parquet
EXPERIMENT_CONCURRENT_THREADS=50
EXPERIMENT_WARMUP_QUERIES=1000
EXPERIMENT_REPETITIONS=5
```

> **Credentials note:** The live OpenSearch cluster credentials will be provided to developers separately and must be added to a local `.env` file. The `.env` file is git-ignored. Never hardcode or commit credentials. If using IAM auth, prefer an EC2 instance profile over explicit access keys — this is the recommended approach for backfill workers running in `us-west-1`.

---

## 11. Python Dependencies

```txt
# requirements.txt

# Qdrant
qdrant-client[grpc]>=1.8.0

# OpenSearch
opensearch-py>=2.4.0
requests-aws4auth>=1.3.0     # IAM-based auth for AOS cluster

# Embedding models
torch>=2.0.0
open-clip-torch>=2.24.0
sentence-transformers>=2.7.0
transformers>=4.38.0

# Data
numpy>=1.26.0
pandas>=2.1.0
pyarrow>=14.0.0          # parquet I/O

# AWS
boto3>=1.34.0
botocore>=1.34.0

# Experiment / statistics
scipy>=1.12.0
scikit-learn>=1.4.0      # brute-force exact KNN reference

# Async
asyncio
aiohttp>=3.9.0
httpx>=0.26.0

# Utilities
python-dotenv>=1.0.0
pydantic>=2.5.0
tqdm>=4.66.0
loguru>=0.7.0
```

```txt
# requirements-dev.txt
pytest>=7.4.0
pytest-asyncio>=0.23.0
pytest-benchmark>=4.0.0
black>=24.0.0
ruff>=0.2.0
mypy>=1.8.0
```

---

## 12. Development Conventions

### 12.1 Code Style

- **Python 3.11+** — use `match` statements, `TypeAlias`, `Self` where appropriate
- **Type hints everywhere** — mypy strict mode; no `Any` without explicit justification
- **Async-first** — all I/O functions are `async def`; use `asyncio.gather` for parallel calls
- **Pydantic models** for all data structures crossing service boundaries
- **loguru** for logging — structured JSON in production, pretty console in dev
- **black** + **ruff** for formatting and linting

### 12.2 Error Handling

```python
# OpenSearch (extant) is READ-ONLY — this codebase never writes to it.
# All write operations target Qdrant only (Stage 1).

# ── Stage 1 error patterns ────────────────────────────────────────
# Ingestion (scroll → embed → Qdrant):
#   OS scroll fail      → raise immediately; do not silently skip
#   Image fetch fail    → log to dead-letter queue; continue with specifics-only point
#   Embedding fail      → log to dead-letter queue; retry with backoff
#   Qdrant write fail   → log to dead-letter queue; reconciliation job re-attempts

# Experiment runner:
#   Qdrant query fail   → mark query as failed; exclude from statistical analysis
#   OS KNN query fail   → mark query as failed; exclude from statistical analysis
#   Do NOT fall back silently — a failed query must be logged, not swallowed,
#   to preserve the integrity of the comparative analysis

# ── Stage 2 error patterns (not yet applicable) ──────────────────
# New OS write fail    → log; retry; do not write to extant OS as fallback
# Migration gap        → track uncopied documents; reconciliation job fills gaps
```

### 12.3 Naming Conventions

| Entity | Convention | Example |
|--------|-----------|---------|
| Qdrant collection | `snake_case` | `cards` |
| Named vectors | `snake_case` | `image`, `specifics` |
| Payload keys | `snake_case` | `os_id`, `has_image` |
| OpenSearch index | `kebab-case` | `cards-v2` |
| S3 prefixes | `kebab-case/` | `embeddings/image/` |
| Python modules | `snake_case` | `qdrant_search.py` |
| Python classes | `PascalCase` | `ImageEncoder` |
| Constants | `UPPER_SNAKE` | `EMBEDDING_DIM` |

### 12.4 Testing Strategy

```
Unit tests:        Pure functions — metrics, statistics, ID conversion, payload building
Integration tests: Live local Qdrant (docker-compose.dev.yml) — collection ops, search
Experiment tests:  Full pipeline on 10k-item sample — validates methodology end-to-end
Performance tests: pytest-benchmark on search and encoding hotpaths
```

```bash
# Run unit tests only (no external dependencies)
pytest tests/unit/

# Run integration tests (requires local Qdrant running)
docker compose -f docker/docker-compose.dev.yml up -d
pytest tests/integration/

# Run experiment on 10k sample (quick validation)
python experiment/runner/run_experiment.py --sample 10000 --dry-run
```

---

## 13. Key Implementation Decisions

### 13.1 Why gRPC over REST

Use `qdrant-client[grpc]` and `prefer_grpc=True`. gRPC is ~30% faster for high-throughput search due to binary serialisation and persistent HTTP/2 connections. Use NLB (not ALB) in production to avoid HTTP/2 proxy overhead.

### 13.2 S3 as Primary Durable Vector Store

S3 is the **primary durable store for all generated vectors**. Qdrant is a queryable index built from those vectors, not the source of truth. This means:

- If Qdrant is lost, corrupted, or needs to be rebuilt with different parameters, vectors are re-loaded from S3 — no re-embedding required.
- If a new embedding model is introduced, the old vectors remain in S3 as a versioned baseline for comparison.
- Vectors are written to S3 **before** being loaded into Qdrant. A vector that exists in Qdrant but not S3 is a pipeline bug.

See **Section 16** for the full S3 vector store specification: bucket structure, key naming convention, file format, write/read APIs, and the repopulation procedure.

**Format:** Parquet (columnar, compressed with Snappy). Parquet allows efficient reads of only the `os_id` and `vector` columns during Qdrant load without deserialising the full file. It also supports predicate pushdown for targeted repopulation (e.g. re-load only eBay image vectors for a given date range).

### 13.3 Why `wait=False` for Backfill, `wait=True` for Production

During backfill, `wait=False` allows the Qdrant server to index asynchronously while the client sends the next batch — roughly 3x throughput improvement. For production writes, `wait=True` ensures the point is queryable immediately after the API call returns.

### 13.4 Why Three Prefetch Arms in Search

- **Arm 1** (image, has_image=True): Captures visual similarity for sold cards and catalogue cards that have images. This is the primary signal for image search.
- **Arm 2** (specifics, all cards): Adds semantic specifics similarity across all document types — critical for card identification.
- **Arm 3** (specifics, has_image=False only): Gives Type 3 catalogue cards (no image) a dedicated candidate pool. Without this arm they are numerically disadvantaged against the larger image-bearing population in Arm 2.

RRF then merges the three ranked lists — candidates appearing in multiple arms receive a higher fused score.

### 13.5 Why Separate Vector Namespace per Embedding Model Version

When upgrading from e.g. `image_vitl14` to `image_vitl14_v2`:
1. Add new named vector field to collection schema (non-breaking, zero downtime)
2. Backfill new vectors on background workers
3. Route 10% of traffic to new vector field (A/B test)
4. Once validated, shift 100% and remove old field

This avoids full re-index and allows progressive validation. See `src/schema/collection.py` for versioning conventions.

### 13.6 Why itemSpecifics Inference Is Deferred to Stage 2

The non-eBay marketplace inference pipeline (Section 17.8) is deliberately excluded from Stage 1. The reason is scope control: the comparative experiment in Stage 1 evaluates the vector search capability of Qdrant versus OpenSearch on a like-for-like basis. Introducing inferred specifics of variable quality during Stage 1 would add a confounding variable to the experiment — it would be impossible to determine whether recall differences were due to the search system or the quality of the inferred specifics.

In Stage 1, non-eBay marketplace documents are ingested into Qdrant with:
- `specifics_source = "none"` (no itemSpecifics available, no inference attempted)
- `specifics_vector` omitted or zeroed
- `image` vector populated if `galleryURL` is present

The inference pipeline is designed and ready in `stage2/src/embeddings/specifics_inferrer.py` but is not activated until Stage 2.

---

## 14. Known Constraints and Gotchas

| Area | Constraint | Mitigation |
|------|-----------|------------|
| Qdrant point IDs | Must be `uint64` or `UUID` — not arbitrary strings | Prefer numeric `id` field (long) as direct uint64; fall back to `uuid5(NAMESPACE, _id)` |
| Live OS endpoint | Single AOS cluster in `us-west-1` is the source of truth and control system | Use env var `OPENSEARCH_HOST`; run backfill workers in same region to avoid cross-region transfer costs |
| Per-date indices | OS uses `YYYY-MM-DD` index names — no single index alias by default | Use `date_index_pattern(start, end)` helper for explicit lists, or `"*"` for full corpus scroll |
| `_source` excludes vectors | `imageVector` and `textVector` are excluded from OS `_source` in the mapping | Re-generate embeddings from `galleryURL` (image) and `itemSpecifics` (text) during backfill |
| `itemSpecifics` sparsity | Not all fields present on every document — some listings omit set, cardNumber, etc. | All payload extraction uses `.get()` with `None` default; `format_specifics()` skips empty parts |
| `lowercase_normalizer` | OS normalises all itemSpecifics keyword fields to lowercase at index time | Apply `.lower().strip()` to all string payload values before writing to Qdrant |
| `cardNumber` OS field name | OS mapping uses `cardNumber` (camelCase); Qdrant payload uses `card_number` (snake_case) | `extract_payload()` handles the rename — always use `card_number` in Qdrant filter code |
| `type` field collision | Both the document type ("sold"/"catalogue") and `itemSpecifics.type` ("rookie"/"base") use "type" | Document type stored as `type`; itemSpecifics type stored as `card_type` in payload |
| OpenSearch `mget` | Max 1,000 IDs per request | Batch large result sets into chunks of 500 |
| CLIP preprocessing | Must be identical at index time and query time | Single shared `get_clip_transform()` function in `image_encoder.py` |
| `galleryURL` as image source | Image embedding uses `galleryURL` (keyword, up to 2048 chars) | Validate URL is reachable before embedding; log failures to dead-letter queue |
| Non-eBay indices lack itemSpecifics | `-gold`, `-pris`, `-heritage`, `-pwcc`, `-ms` indices have no structured card attributes | Use `specifics_inferrer.py` to infer from title; always set `specifics_source` in payload |
| Inferred specifics lower confidence | Auto-generated itemSpecifics may be wrong, especially for rare or unusual cards | Apply `INFERRED_SPECIFICS_BOOST_MULTIPLIER = 0.5` in fusion; flag in UI where shown |
| Non-eBay index names unknown at build time | Suffix-based index names may include a variable base (e.g. `auction-2024-gold`) | Use `discover_indices()` + `classify_index()` at runtime; never hardcode index names |
| `wait=False` backfill | Qdrant may not have indexed points immediately | Don't query during active backfill; use collection status API to check indexing progress |
| Payload filter + ANN | Qdrant maintains recall under filtering; OpenSearch does not | Do not apply OS post-filters to KNN results in experiment — measure degradation honestly |
| S3 write before Qdrant | Vector must exist in S3 before Qdrant upsert — never load into Qdrant without persisting to S3 first | `vector_store.py` enforces this order; pipeline raises if S3 write fails before attempting Qdrant upsert |
| S3 key uniqueness | Two runs with same model + params must produce the same key prefix to enable deduplication | Key is fully deterministic from model name, version, params, and os_id — no timestamps in object keys |
| r6gd NVMe | Instance store is ephemeral — lost on stop/terminate | S3 is the durable store; Qdrant is rebuilt from S3 on NVMe loss via repopulate command |
| gRPC TLS | AWS NLB requires TLS passthrough mode for gRPC | Use `ssl_ca_cert` parameter in QdrantClient if TLS is enabled end-to-end |
| INT8 quantization | Slight recall degradation for hard queries | `rescore=True` recovers most of this; increase `hnsw_ef` to 256 for hard queries |

---

## 15. Useful Commands

```bash
# Start local Qdrant
docker compose -f docker/docker-compose.dev.yml up -d

# Check Qdrant cluster health
curl http://localhost:6333/healthz
curl http://localhost:6333/collections/cards

# Verify live OpenSearch cluster is reachable (read-only health check)
curl -u "$OPENSEARCH_USER:$OPENSEARCH_PASSWORD"   "https://$OPENSEARCH_HOST/_cluster/health?pretty"

# List all date indices on the live cluster
curl -u "$OPENSEARCH_USER:$OPENSEARCH_PASSWORD"   "https://$OPENSEARCH_HOST/_cat/indices?v&s=index"

# Count documents across all date indices
curl -u "$OPENSEARCH_USER:$OPENSEARCH_PASSWORD"   "https://$OPENSEARCH_HOST/*/_count?pretty"

# Create collection (local dev)
python -c "from src.schema.collection import create_collection; create_collection()"

# Run full embedding backfill locally (small sample)
# Vectors are always written to S3 first, then loaded into Qdrant
python src/embeddings/batch_job.py \
  --index-pattern "2024-01-15" \
  --batch-size 1000 \
  --s3-bucket $S3_VECTOR_BUCKET \
  --s3-prefix $S3_VECTOR_PREFIX \
  --dry-run   # remove --dry-run to actually write

# Repopulate Qdrant from existing S3 vectors (no re-embedding)
python src/embeddings/vector_store.py repopulate \
  --s3-bucket $S3_VECTOR_BUCKET \
  --s3-prefix $S3_VECTOR_PREFIX \
  --vector-type image \
  --model clip-vit-l-14 \
  --index-pattern "2024-*"

# List all vector files in S3 for a given model and type
python src/embeddings/vector_store.py list \
  --s3-bucket $S3_VECTOR_BUCKET \
  --s3-prefix $S3_VECTOR_PREFIX \
  --vector-type image \
  --model clip-vit-l-14

# Run experiment (full)
python experiment/runner/run_experiment.py \
  --query-set experiment/dataset/query_set.parquet \
  --ground-truth experiment/dataset/ground_truth.parquet \
  --output experiment/results/$(date +%Y%m%d)/

# Generate statistical report
python experiment/analysis/report_generator.py \
  --results experiment/results/latest/ \
  --output experiment/results/latest/report.json

# Qdrant snapshot (production backup)
curl -X POST "http://qdrant-host:6333/collections/cards/snapshots" \
  -H "api-key: $QDRANT_API_KEY"
```

---

## 18. OpenSearch Index Mapping (Source of Truth)

The production OpenSearch cluster is **live and accessible** at the endpoint below. It uses **per-date indices** named `YYYY-MM-DD` and holds the full sold listing dataset including `imageVector` and `textVector` fields used for KNN search. This cluster is the **control system** in the experiment and the **source of data** for the Qdrant backfill.

### 17.0 Live Cluster Endpoint

```
Host:     https://search-es130point-vector-h3eaau7mwcpyhgynkthfcwzeje.aos.us-west-1.on.aws
Region:   us-west-1
Service:  Amazon OpenSearch Service (AOS)
Auth:     Credentials supplied via .env (see Section 10)
```

```python
# src/ingestion/opensearch_reader.py — connection setup
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import boto3, os

def get_opensearch_client() -> OpenSearch:
    """
    Connects to the live AOS cluster.
    Supports both IAM-based auth (AWS4Auth) and basic auth —
    whichever credentials are present in the environment.
    """
    host = os.environ["OPENSEARCH_HOST"]  # without https:// prefix
    use_iam = os.environ.get("OPENSEARCH_USE_IAM", "false").lower() == "true"

    if use_iam:
        credentials = boto3.Session().get_credentials()
        auth = AWS4Auth(
            credentials.access_key,
            credentials.secret_key,
            os.environ["AWS_REGION"],
            "es",
            session_token=credentials.token,
        )
    else:
        auth = (
            os.environ["OPENSEARCH_USER"],
            os.environ["OPENSEARCH_PASSWORD"],
        )

    return OpenSearch(
        hosts=[{"host": host, "port": 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=60,
    )
```

> **Important — AWS region:** The cluster is in `us-west-1`. If running ingestion or experiment workers from a different region, cross-region data transfer costs apply (~$0.02/GB). Run backfill workers in `us-west-1` to avoid this.

> **CRITICAL — READ-ONLY:** The extant OpenSearch cluster is a **read-only data source** for this project. No code in this repository may create, update, or delete any index, mapping, document, or setting on this cluster under any circumstances — including during development, testing, or experiment runs. The cluster is managed by an upstream pipeline that is entirely outside the scope of this codebase. Any accidental write to this cluster risks corrupting production data.

### 17.1 Index Taxonomy

The cluster contains indices from **six different marketplaces**, each with a distinct naming convention and data completeness profile. Understanding this taxonomy is essential for the scroll reader, payload extraction, and the itemSpecifics inference pipeline.

#### 17.1.1 eBay Indices — `YYYY-MM-DD`

```
Format:   YYYY-MM-DD  (e.g. 2024-03-15)
Source:   eBay (global — all sites identified by globalId field)
Includes: Full itemSpecifics object with all structured fields
          imageVector and textVector (excluded from _source)
          galleryURL for image embedding source
```

These are the **highest quality indices**. itemSpecifics are provided directly by eBay sellers and cover all card identity fields (`set`, `cardNumber`, `player`, `grade`, etc.). They are the ground truth for card identification and the primary training data for the itemSpecifics inference model used on other marketplace indices.

#### 17.1.2 Non-eBay Marketplace Indices

All non-eBay indices use a suffix appended to a base identifier. None of these indices include an `itemSpecifics` field in their mapping — structured card attributes must be inferred from the `title` field.

| Suffix | Marketplace | Full Name | Notes |
|--------|-------------|-----------|-------|
| `-gold` | Goldin Auctions | Goldin | High-value vintage and modern cards |
| `-pris` | Pristine Auctions | Pristine Auctions | Graded card specialist |
| `-heritage` | Heritage Auctions | Heritage Auctions | Vintage/rare cards, memorabilia |
| `-pwcc` | Fanatics Collect | Formerly PWCC Marketplace | High-volume graded card auctions |
| `-ms` | MySlabs | MySlabs | Graded card marketplace |

> **Index name format:** The base identifier for non-eBay indices is not date-based. Use `_cat/indices` on the live cluster to discover the exact current index names for each suffix.

#### 17.1.3 Index Discovery

```python
# src/ingestion/opensearch_reader.py — index discovery helpers

EBAY_INDEX_PATTERN    = r"^\d{4}-\d{2}-\d{2}$"          # YYYY-MM-DD exactly
NON_EBAY_SUFFIXES     = ["-gold", "-pris", "-heritage", "-pwcc", "-ms"]

MARKETPLACE_MAP = {
    "-gold":     "goldin",
    "-pris":     "pristine",
    "-heritage": "heritage",
    "-pwcc":     "fanatics_collect",   # formerly PWCC
    "-ms":       "myslabs",
}

import re

def classify_index(index_name: str) -> dict:
    """
    Classify an index by its naming convention.
    Returns dict with 'marketplace', 'has_item_specifics', and 'index_type'.
    """
    if re.match(EBAY_INDEX_PATTERN, index_name):
        return {
            "marketplace":        "ebay",
            "has_item_specifics": True,
            "index_type":         "ebay_dated",
        }
    for suffix, marketplace in MARKETPLACE_MAP.items():
        if index_name.endswith(suffix):
            return {
                "marketplace":        marketplace,
                "has_item_specifics": False,
                "index_type":         "marketplace_suffix",
            }
    return {
        "marketplace":        "unknown",
        "has_item_specifics": False,
        "index_type":         "unknown",
    }

def discover_indices(client: OpenSearch) -> list[dict]:
    """
    List all indices on the cluster and classify them.
    Returns sorted list of dicts with name, doc_count, and classification.
    """
    response = client.cat.indices(format="json", h="index,docs.count,store.size")
    return [
        {**row, **classify_index(row["index"])}
        for row in response
        if not row["index"].startswith(".")   # exclude system indices
    ]
```

---

### 17.2 Index Settings

```json
{
  "settings": {
    "index": {
      "knn": true,
      "number_of_shards": 1,
      "number_of_replicas": 2
    },
    "analysis": {
      "normalizer": {
        "lowercase_normalizer": {
          "type": "custom",
          "filter": ["lowercase", "asciifolding"]
        }
      }
    }
  }
}
```

> **Note — `knn: true`:** This setting enables the OpenSearch k-NN plugin. The live cluster actively serves KNN queries against `imageVector` and `textVector` — this is the current production search path and the control condition in the experiment. After migration, new indices created for document-only storage should set `knn: false`.

### 17.3 Source Exclusions and Vector Behaviour and Vector Behaviour

```json
"_source": {
  "excludes": ["imageVector", "textVector"]
}
```

`imageVector` and `textVector` are stored in the Lucene index for KNN traversal but **excluded from `_source`**. This applies to **all index types** (eBay dated and marketplace suffix). Non-eBay indices additionally lack `itemSpecifics` in `_source` — see Section 17.1.2. This has two distinct implications:

**For KNN search (what the live cluster DOES support):**
- `imageVector` and `textVector` ARE used internally by the k-NN plugin to perform HNSW graph traversal.
- KNN queries (`"query": {"knn": {"imageVector": {...}}}`) work correctly against the live cluster — this is the control system query path.
- The experiment runner uses this to obtain OpenSearch baseline results.

**For data retrieval (what is NOT possible):**
- Vectors **cannot be retrieved** via `GET`, `search`, `scroll`, or `mget` — they are not in `_source`.
- During Qdrant backfill, embeddings must be **re-generated** from `galleryURL` (image via CLIP) and `itemSpecifics` (text via MiniLM). They cannot be copied directly from OpenSearch.
- This project never writes to OpenSearch. Qdrant owns all vectors. OpenSearch is treated as an immutable read-only data source throughout.

### 17.4 Top-Level Fields

| OS Field | OS Type | Qdrant Payload | Notes |
|----------|---------|----------------|-------|
| `id` | `long` | `os_id` (also used as Qdrant uint64 point ID) | Primary numeric identifier |
| `itemId` | `text` | `item_id` | eBay item ID string (not analysed for KV lookup) |
| `source` | `keyword` | `source` | e.g. `"ebay_uk"`, `"ebay_us"` |
| `globalId` | `keyword` | `global_id` | eBay site: `"EBAY-GB"`, `"EBAY-US"` |
| `title` | `text` + `search_as_you_type` + `keyword` | **OpenSearch only** | Full-text search; not in Qdrant payload |
| `galleryURL` | `keyword` (ignore_above 2048) | **Used for image embedding** | Source URL for CLIP encoder; not in payload |
| `itemURL` | `keyword` (ignore_above 2048) | **OpenSearch only** | Listing URL; display field only |
| `saleType` | `keyword` | `sale_type` | `"FixedPrice"` \| `"Auction"` |
| `currentPrice` | `double` | `current_price` | Snapshot at index time |
| `currentPriceCurrency` | `keyword` | `currency` | ISO 4217 currency code |
| `endTime` | `date` | **OpenSearch only** | Date range queries stay in OS |
| `bidCount` | `integer` | **OpenSearch only** | Aggregation field |
| `salePrice` | `double` | **OpenSearch only** | Final sale price |
| `shippingServiceCost` | `double` | **OpenSearch only** | |
| `BestOfferPrice` | `double` | **OpenSearch only** | |
| `BestOfferCurrency` | `keyword` | **OpenSearch only** | |

### 17.5 itemSpecifics Fields (eBay Indices Only)

> These fields are present **only in `YYYY-MM-DD` eBay indices**. Non-eBay marketplace indices (`-gold`, `-pris`, `-heritage`, `-pwcc`, `-ms`) do not include `itemSpecifics` in their mapping. For those indices, structured specifics must be inferred — see Section 17.8.

All `itemSpecifics` fields use `lowercase_normalizer` — values are lowercased and ASCII-folded at index time. All values written to Qdrant payload must be lowercased to match.

| OS Field (`itemSpecifics.*`) | OS Type | Qdrant Payload Key | Used For |
|-------------------------------|---------|-------------------|----------|
| `brand` | `keyword` | `brand` | Card manufacturer (e.g. `"topps"`, `"panini"`) |
| `player` | `keyword` | `player` | Athlete or character name — **highest identity signal** |
| `genre` | `keyword` | `genre` | Sport or game (e.g. `"football"`, `"pokemon"`) — used for stratification |
| `country` | `keyword` | `country` | Country of origin / print region |
| `set` | `keyword` | `set` | Card set name — **primary near-dup filter field** |
| `cardNumber` | `keyword` | `card_number` | Card number within set (e.g. `"4/102"`) — **exact identity with `set`** |
| `subset` | `keyword` | `subset` | Insert or parallel set name |
| `parallel` | `keyword` | `parallel` | Parallel variant (e.g. `"holo"`, `"gold refractor"`) |
| `serialNumber` | `keyword` | `serial_number` | Serial number for numbered cards |
| `year` | `integer` | `year` | Release year |
| `graded` | `boolean` | `graded` | Whether the card is graded |
| `grader` | `keyword` | `grader` | Grading company (e.g. `"psa"`, `"bgs"`, `"cgc"`) |
| `grade` | `keyword` | `grade` | Grade value (e.g. `"9"`, `"9.5"`, `"10"`) |
| `type` | `keyword` | `card_type` | Card type (e.g. `"rookie"`, `"base"`, `"insert"`) — **renamed in payload** to avoid collision with document type field |
| `autographed` | `boolean` | `autographed` | Auto / signed card |
| `team` | `keyword` | `team` | Team name (sports cards) |

### 17.6 Card Identity Logic

Card identification uses a **hierarchical specificity model** based on `itemSpecifics` fields:

```
Level 1 — Exact match (relevance score 3):
  set + cardNumber + grader + grade + parallel
  → Identifies the specific graded copy / variant

Level 2 — Correct match (relevance score 2):
  set + cardNumber
  → Identifies the specific card regardless of condition

Level 3 — Related (relevance score 1):
  player + genre OR brand + set (without cardNumber)
  → Same player/set but wrong card

Level 4 — Irrelevant (relevance score 0):
  No meaningful overlap
```

This hierarchy drives both the **ground truth relevance annotations** for nDCG and the **Qdrant score boost policy** for catalogue cards.

### 17.7 Qdrant Filter Patterns

```python
# src/search/qdrant_search.py — filter builder
# All values must be pre-lowercased before passing to MatchValue

from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

def build_filter(criteria: dict) -> Filter | None:
    """
    Build a Qdrant Filter from a dict of itemSpecifics criteria.
    Keys map to Qdrant payload field names (snake_case).
    Values are pre-lowercased strings, booleans, or numeric ranges.

    Examples:
        {"set": "base set", "graded": True}
        {"genre": "pokemon", "grader": "psa", "grade": "10"}
        {"player": "michael jordan", "year": 1986}
    """
    if not criteria:
        return None

    conditions = []
    for key, value in criteria.items():
        if value is None:
            continue
        if isinstance(value, bool):
            conditions.append(
                FieldCondition(key=key, match=MatchValue(value=value))
            )
        elif isinstance(value, (int, float)):
            conditions.append(
                FieldCondition(key=key, match=MatchValue(value=value))
            )
        elif isinstance(value, str):
            conditions.append(
                FieldCondition(key=key, match=MatchValue(value=value.lower().strip()))
            )
        elif isinstance(value, dict) and ("gte" in value or "lte" in value):
            # Range filter — used for year, current_price
            conditions.append(
                FieldCondition(key=key, range=Range(**value))
            )

    return Filter(must=conditions) if conditions else None


# Common filter examples for card identification:
EXAMPLE_FILTERS = {
    # Narrow to specific set to suppress near-duplicates
    "by_set":           {"set": "base set"},
    # Graded PSA 10 only
    "psa_10":           {"graded": True, "grader": "psa", "grade": "10"},
    # All graded cards in a set
    "graded_in_set":    {"set": "base set", "graded": True},
    # Sports cards for a specific player
    "player_filter":    {"player": "michael jordan", "genre": "basketball"},
    # UK eBay source only
    "uk_source":        {"global_id": "ebay-gb"},
}
```

---

### 17.8 Non-eBay Marketplace Indices — itemSpecifics Inference (Stage 2 Only)

> **This section describes Stage 2 functionality.** The inference pipeline is designed here for planning purposes but is **not implemented or activated in Stage 1**. In Stage 1, non-eBay documents are loaded into Qdrant with `specifics_source = "none"` and no specifics vector. See Section 13.6 for the rationale.
>
> Implementation lives in `stage2/src/embeddings/specifics_inferrer.py` — not in `src/`.

Non-eBay marketplace indices (`-gold`, `-pris`, `-heritage`, `-pwcc`, `-ms`) do not carry an `itemSpecifics` field. The only textual identity signal available is the `title` field. To enable consistent card identification and Qdrant payload population across all marketplaces, structured `itemSpecifics` must be **inferred from the title** and flagged as auto-generated.

#### 17.8.1 Inference Strategy

The inference pipeline uses a two-step approach:

```
title (raw string from non-eBay index)
    │
    ▼ Step 1: Rule-based extraction
    │  Regex + lookup tables extract high-confidence fields
    │  (grader, grade, year, serial number patterns)
    │
    ▼ Step 2: Semantic matching against eBay itemSpecifics
    │  Encode title with MiniLM → search Qdrant specifics index
    │  Find nearest eBay sold listings → harvest their itemSpecifics
    │  Majority-vote across top-K matches for each field
    │
    ▼ Output: inferred itemSpecifics dict + confidence scores
```

#### 17.8.2 Payload Flag

Every Qdrant point ingested from a non-eBay source **must** carry provenance metadata so downstream consumers can distinguish eBay-provided specifics from inferred ones:

```python
# src/ingestion/qdrant_writer.py — provenance fields added to PAYLOAD_SYSTEM

PAYLOAD_PROVENANCE = {
    # Source of itemSpecifics data:
    #   "ebay"      — provided directly by the eBay listing (highest confidence)
    #   "inferred"  — generated by the inference pipeline from title text
    #   "none"      — no itemSpecifics available and inference was not attempted
    "specifics_source":     str,

    # Confidence score for inferred specifics (0.0–1.0).
    # None / omitted when specifics_source == "ebay".
    "specifics_confidence": float | None,

    # Qdrant point IDs of the eBay listings used as inference reference.
    # Populated when specifics_source == "inferred". Allows auditing/retraining.
    "specifics_ref_ids":    list[str] | None,

    # Name of the inference model version used.
    # Populated when specifics_source == "inferred".
    "specifics_model":      str | None,
}
```

> **Search behaviour impact:** The `specifics_source` field is stored in Qdrant payload and can be used at query time to down-weight inferred specifics in the scoring pipeline (see `src/search/fusion.py`). Inferred specifics should be treated as lower-confidence than eBay-provided ones and should **not** trigger the full catalogue boost policy — apply a reduced boost multiplier (e.g. 0.5×) when `specifics_source == "inferred"`.

#### 17.8.3 Inference Implementation

```python
# src/embeddings/specifics_inferrer.py

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
import re
from collections import Counter

# ── Step 1: Rule-based field extraction ──────────────────────────
GRADER_PATTERNS = {
    "psa":  r"\bpsa\b",
    "bgs":  r"\bbgs\b|\bbeckett\b",
    "cgc":  r"\bcgc\b",
    "sgc":  r"\bsgc\b",
    "hga":  r"\bhga\b",
}
GRADE_PATTERN   = r"(?:grade[d]?\s*)?(\d+(?:\.\d+)?)\s*(?:/10)?"
YEAR_PATTERN    = r"(19[5-9]\d|20[0-2]\d)"
SERIAL_PATTERN  = r"(\d+)\s*/\s*(\d+)"   # e.g. "42/99"

def extract_rules(title: str) -> dict:
    """Extract high-confidence fields from title using regex patterns."""
    t = title.lower()
    result = {}

    for grader, pattern in GRADER_PATTERNS.items():
        if re.search(pattern, t):
            result["grader"] = grader
            result["graded"] = True
            break

    if grade_match := re.search(GRADE_PATTERN, t):
        result["grade"] = grade_match.group(1)

    if year_match := re.search(YEAR_PATTERN, t):
        result["year"] = int(year_match.group(1))

    if serial_match := re.search(SERIAL_PATTERN, t):
        result["serial_number"] = f"{serial_match.group(1)}/{serial_match.group(2)}"

    return result


# ── Step 2: Semantic matching via Qdrant ─────────────────────────
INFERENCE_TOP_K       = 10    # eBay listings to retrieve for voting
INFERENCE_MIN_VOTES   = 3     # minimum votes for a field value to be accepted
INFERENCE_MIN_CONF    = 0.60  # minimum confidence to include a field in output

def infer_from_qdrant(
    title: str,
    qdrant_client: QdrantClient,
    text_encoder,
    collection: str = "cards",
) -> tuple[dict, float, list[str]]:
    """
    Infer itemSpecifics for a non-eBay listing by:
      1. Encoding the title as a specifics vector
      2. Finding nearest eBay listings in Qdrant
      3. Majority-voting their itemSpecifics fields

    Returns: (inferred_specifics, confidence_score, ref_point_ids)
    """
    title_vec = text_encoder.encode(title.lower())

    results = qdrant_client.search(
        collection_name=collection,
        query_vector=("specifics", title_vec.tolist()),
        query_filter=Filter(must=[
            FieldCondition(key="source", match=MatchValue(value="ebay")),
            FieldCondition(key="specifics_source", match=MatchValue(value="ebay")),
        ]),
        limit=INFERENCE_TOP_K,
        with_payload=True,
    )

    if not results:
        return {}, 0.0, []

    ref_ids = [str(r.id) for r in results]

    # Vote across retrieved payloads for each specifics field
    VOTABLE_FIELDS = ["player", "brand", "genre", "set", "card_number",
                      "subset", "parallel", "team", "country", "card_type"]
    votes: dict[str, Counter] = {f: Counter() for f in VOTABLE_FIELDS}

    for hit in results:
        p = hit.payload
        for field in VOTABLE_FIELDS:
            val = p.get(field)
            if val:
                votes[field][val] += 1

    inferred = {}
    field_confidences = []

    for field, counter in votes.items():
        if not counter:
            continue
        top_val, top_count = counter.most_common(1)[0]
        conf = top_count / INFERENCE_TOP_K
        if top_count >= INFERENCE_MIN_VOTES and conf >= INFERENCE_MIN_CONF:
            inferred[field] = top_val
            field_confidences.append(conf)

    overall_confidence = (
        sum(field_confidences) / len(field_confidences)
        if field_confidences else 0.0
    )

    return inferred, overall_confidence, ref_ids


def infer_specifics(
    title: str,
    qdrant_client: QdrantClient,
    text_encoder,
    collection: str = "cards",
) -> dict:
    """
    Full inference pipeline: rule-based extraction + semantic Qdrant matching.
    Returns a dict suitable for use as itemSpecifics in extract_payload(),
    plus provenance metadata.
    """
    rule_based   = extract_rules(title)
    semantic, confidence, ref_ids = infer_from_qdrant(
        title, qdrant_client, text_encoder, collection
    )

    # Rule-based results take precedence over semantic (higher confidence)
    merged = {**semantic, **rule_based}

    return {
        "inferred_specifics":   merged,
        "specifics_source":     "inferred" if merged else "none",
        "specifics_confidence": round(confidence, 3),
        "specifics_ref_ids":    ref_ids,
        "specifics_model":      "minilm-l6-v2-qdrant-vote-v1",
    }
```

#### 17.8.4 Payload Population for Non-eBay Sources

```python
# src/ingestion/qdrant_writer.py — non-eBay extract_payload variant

def extract_payload_non_ebay(os_hit: dict, inferred: dict) -> dict:
    """
    Build Qdrant payload for a non-eBay marketplace document.
    itemSpecifics are replaced with inferred values; provenance is flagged.

    os_hit:   raw OpenSearch scroll hit from a -gold / -pris / -heritage / -pwcc / -ms index
    inferred: output from infer_specifics() — includes inferred_specifics + provenance
    """
    src   = os_hit["_source"]
    specs = inferred.get("inferred_specifics", {})

    def lower(val) -> str:
        return str(val).lower().strip() if val is not None else ""

    return {
        # Core identity
        "os_id":        os_hit["_id"],
        "item_id":      str(src.get("itemId", "")),
        "type":         "sold",
        "has_image":    bool(src.get("galleryURL")),
        "source":       lower(src.get("source")),
        "global_id":    lower(src.get("globalId")),
        "active":       True,

        # itemSpecifics — inferred from title, NOT from the OS document
        "brand":         lower(specs.get("brand")),
        "player":        lower(specs.get("player")),
        "genre":         lower(specs.get("genre")),
        "country":       lower(specs.get("country")),
        "set":           lower(specs.get("set")),
        "card_number":   lower(specs.get("card_number")),
        "subset":        lower(specs.get("subset")),
        "parallel":      lower(specs.get("parallel")),
        "serial_number": lower(specs.get("serial_number")),
        "year":          int(specs["year"]) if specs.get("year") else None,
        "graded":        bool(specs.get("graded", False)),
        "grader":        lower(specs.get("grader")),
        "grade":         lower(specs.get("grade")),
        "card_type":     lower(specs.get("card_type")),
        "autographed":   bool(specs.get("autographed", False)),
        "team":          lower(specs.get("team")),

        # Sale context
        "sale_type":     lower(src.get("saleType")),
        "current_price": float(src["currentPrice"]) if src.get("currentPrice") else None,
        "currency":      lower(src.get("currentPriceCurrency")),

        # Provenance — always populated for non-eBay sources
        "specifics_source":     inferred.get("specifics_source", "none"),
        "specifics_confidence": inferred.get("specifics_confidence"),
        "specifics_ref_ids":    inferred.get("specifics_ref_ids"),
        "specifics_model":      inferred.get("specifics_model"),

        # System
        "image_is_pseudo":           False,
        "pseudo_vector_sample_size": 0,
        "indexed_at":                datetime.utcnow().isoformat(),
    }
```

#### 17.8.5 Ingestion Pipeline Routing

The pipeline in `src/ingestion/pipeline.py` uses `classify_index()` to route each document. The routing logic differs by stage:

```python
# src/ingestion/pipeline.py
# DEVELOPMENT_STAGE env var controls which branch is active.
# Stage 1: non-eBay docs get image vector only; no inference; specifics_source="none"
# Stage 2: non-eBay docs go through inference pipeline in stage2/

import os
from src.ingestion.opensearch_reader import classify_index
from src.ingestion.qdrant_writer import extract_payload

STAGE = int(os.environ.get("DEVELOPMENT_STAGE", "1"))

async def process_hit(os_hit: dict, index_name: str, clients: dict) -> PointStruct | None:
    """Route a single OS hit through the correct extraction path for current stage."""
    classification = classify_index(index_name)
    src = os_hit["_source"]

    # Image embedding — same for all index types and stages
    image_vec = await clients["image_encoder"].encode(src.get("galleryURL"))

    if classification["has_item_specifics"]:
        # eBay index — real itemSpecifics available in both stages
        specifics_text = format_specifics(src.get("itemSpecifics", {}))
        specifics_vec  = clients["text_encoder"].encode(specifics_text)
        payload        = extract_payload(os_hit)
        payload["specifics_source"]     = "ebay"
        payload["specifics_confidence"] = None
        payload["specifics_ref_ids"]    = None
        payload["specifics_model"]      = None
        vectors = {"image": image_vec, "specifics": specifics_vec}

    elif STAGE == 1:
        # Stage 1: non-eBay — store image vector only, no specifics inference
        payload = extract_payload(os_hit)
        payload["specifics_source"]     = "none"
        payload["specifics_confidence"] = None
        payload["specifics_ref_ids"]    = None
        payload["specifics_model"]      = None
        vectors = {"image": image_vec}   # specifics vector intentionally omitted

    else:
        # Stage 2: non-eBay — inference pipeline (imported from stage2/)
        # This branch is unreachable in Stage 1 (DEVELOPMENT_STAGE=1)
        from stage2.src.embeddings.specifics_inferrer import infer_specifics
        from stage2.src.ingestion.qdrant_writer import extract_payload_non_ebay
        title    = src.get("title", "")
        inferred = infer_specifics(title, clients["qdrant"], clients["text_encoder"])
        specifics_text = format_specifics(inferred.get("inferred_specifics", {}))
        specifics_vec  = clients["text_encoder"].encode(specifics_text)
        payload        = extract_payload_non_ebay(os_hit, inferred)
        vectors = {"image": image_vec, "specifics": specifics_vec}

    return PointStruct(
        id=os_id_to_qdrant_id(os_hit["_id"]),
        vector=vectors,
        payload=payload,
    )
```

---

## 19. Stage 2 Planning Notes (Do Not Build Until Stage 1 Approved)

This section captures the design intent for Stage 2 so that decisions are documented while the context is fresh. It is reference material only — no code in `src/` should implement any of this.

### 18.1 New OpenSearch Index Design Principles

The new OpenSearch cluster will differ from the extant cluster in the following ways:

| Aspect | Extant Cluster | New Cluster (Stage 2) |
|--------|---------------|----------------------|
| KNN / vectors | `knn: true`, vectors in index | `knn: false`, no vectors at all |
| Index naming | `YYYY-MM-DD` (eBay) + suffix-based (others) | Single unified index or alias (TBD) |
| itemSpecifics | eBay indices only | All marketplaces (eBay real + others inferred, flagged) |
| Managed by | Upstream pipeline (external) | This codebase (Stage 2 writer) |
| Populated from | External data feeds | Extant cluster via scroll migration |

The exact index structure for the new cluster will be defined in `stage2/docs/new_os_index_design.md` during Stage 2 planning. The extant mapping (Section 17) is the starting point, but the new mapping should be simplified and consolidated.

### 18.2 Migration Pipeline Design

The Stage 2 migration pipeline scrolls the extant cluster and writes to both the new OpenSearch cluster and Qdrant simultaneously:

```
for each index in extant cluster:
    for each document in scroll:
        1. Write full document to new OpenSearch (minus imageVector/textVector)
        2. If document already in Qdrant (from Stage 1): skip Qdrant write
           If document NOT in Qdrant (new data since Stage 1 backfill): add to Qdrant
        3. If non-eBay index: run inference pipeline; store result in both systems
```

Idempotency is critical — the migration must be re-runnable without creating duplicates.

### 18.3 Stage 2 Success Gate

Stage 2 begins only when all of the following are confirmed:

- [ ] Stage 1 experimental report produced and reviewed
- [ ] H₁, H₂, H₃ null hypotheses rejected with large effect sizes (r ≥ 0.50)
- [ ] H₄ equivalence confirmed at Recall@10 ≥ 0.90 (with ef_search=256 remediation applied if needed)
- [ ] Stakeholder sign-off on experimental results
- [ ] Stage 2 scope, timeline, and resource allocation approved
- [ ] New OpenSearch index design reviewed and approved

---

---

## 16. S3 Vector Store

S3 is the **primary durable store for all generated vectors**. Every vector produced by the embedding pipeline is written to S3 before being loaded into Qdrant. Qdrant is treated as a **queryable index** built from S3, not a source of truth. The S3 store supports:

- **Repopulation** — rebuild Qdrant from S3 at any time without re-embedding
- **Model versioning** — old and new model vectors coexist in S3 under different key prefixes
- **Partial reload** — reload vectors for a specific index pattern, model, or date range
- **Audit** — full provenance of every vector (what model, what params, what source document)

---

### 16.1 Bucket Structure

All vectors use a **single S3 bucket** with a structured key namespace. The key encodes all metadata needed to identify, filter, and reload vectors without reading the file contents.

```
s3://{S3_VECTOR_BUCKET}/{S3_VECTOR_PREFIX}/
│
├── image/
│   └── {model_id}/
│       └── {params_hash}/
│           └── {index_type}/
│               └── {partition}/
│                   └── {shard}.parquet
│
├── specifics/
│   └── {model_id}/
│       └── {params_hash}/
│           └── {index_type}/
│               └── {partition}/
│                   └── {shard}.parquet
│
└── checkpoints/
    └── {job_id}.json
```

**Example keys:**

```
vectors/image/clip-vit-l-14/v1-fp16-224px/ebay-dated/2024-01-15/0000.parquet
vectors/image/clip-vit-l-14/v1-fp16-224px/ebay-dated/2024-01-15/0001.parquet
vectors/image/clip-vit-l-14/v1-fp16-224px/gold/all/0000.parquet
vectors/specifics/minilm-l6-v2/v1-default/ebay-dated/2024-01-15/0000.parquet
vectors/specifics/minilm-l6-v2/v1-default/gold/all/0000.parquet
checkpoints/job-20260315-143022-ebay-2024-01.json
```

---

### 16.2 Key Component Definitions

| Component | Description | Values / Examples |
|-----------|-------------|-------------------|
| `{S3_VECTOR_PREFIX}` | Root prefix (from env) | `vectors` |
| `image` / `specifics` | Vector type | Fixed: `image` or `specifics` |
| `{model_id}` | Canonical model identifier | `clip-vit-l-14`, `clip-vit-b-32`, `minilm-l6-v2` |
| `{params_hash}` | Short descriptor of encoding params | `v1-fp16-224px`, `v1-default` (see 16.3) |
| `{index_type}` | Source index classification | `ebay-dated`, `gold`, `pris`, `heritage`, `pwcc`, `ms` |
| `{partition}` | Date for eBay indices; `all` for others | `2024-01-15`, `2024-01`, `all` |
| `{shard}` | Zero-padded shard number within partition | `0000`, `0001`, … |

> **Rule:** Keys are fully deterministic. Given the same model, params, index type, and partition, the key is always identical. This enables idempotent re-runs — a re-run overwrites the same key rather than creating a duplicate.

---

### 16.3 Parameter Hash Convention

The `{params_hash}` component is a **human-readable short descriptor** (not a hash digest) that captures the parameters that affect vector values. If any parameter changes, a new prefix is used — the old vectors remain in S3 for comparison.

```python
# src/embeddings/vector_store.py

# Image encoding params descriptor:
# Format: v{version}-{precision}-{input_size}px
# Examples:
IMAGE_PARAMS = {
    "clip-vit-l-14": {
        "version":    "v1",
        "precision":  "fp16",    # fp32 | fp16 (torch.cuda.amp.autocast)
        "input_size": "224",     # CLIP standard input resolution in pixels
        "normalised": True,      # L2-normalised before storage
    }
}
# → params_hash: "v1-fp16-224px"

# Text encoding params descriptor:
# Format: v{version}-{pooling}-{max_seq}tok
# Examples:
SPECIFICS_PARAMS = {
    "minilm-l6-v2": {
        "version":   "v1",
        "pooling":   "mean",     # mean | cls
        "max_seq":   "256",      # max sequence length in tokens
        "normalised": True,
    }
}
# → params_hash: "v1-mean-256tok"

def build_params_hash(vector_type: str, model_id: str) -> str:
    """Construct the params_hash string for the given model."""
    if vector_type == "image":
        p = IMAGE_PARAMS[model_id]
        return f"{p['version']}-{p['precision']}-{p['input_size']}px"
    elif vector_type == "specifics":
        p = SPECIFICS_PARAMS[model_id]
        return f"{p['version']}-{p['pooling']}-{p['max_seq']}tok"
    raise ValueError(f"Unknown vector_type: {vector_type}")
```

---

### 16.4 Parquet File Schema

Each `.parquet` file within a shard contains the following columns:

```python
# Schema for all vector parquet files
PARQUET_SCHEMA = pa.schema([
    pa.field("os_id",         pa.string()),       # OpenSearch document _id (join key)
    pa.field("qdrant_id",     pa.string()),        # Qdrant point ID (uuid5 of os_id)
    pa.field("index_name",    pa.string()),        # Source OS index (e.g. "2024-01-15", "auction-gold")
    pa.field("index_type",    pa.string()),        # "ebay-dated" | "gold" | "pris" | etc.
    pa.field("vector",        pa.list_(pa.float32())),  # The embedding vector
    pa.field("vector_type",   pa.string()),        # "image" | "specifics"
    pa.field("model_id",      pa.string()),        # e.g. "clip-vit-l-14"
    pa.field("params_hash",   pa.string()),        # e.g. "v1-fp16-224px"
    pa.field("created_at",    pa.timestamp("ms")), # UTC timestamp of embedding creation
    pa.field("source_url",    pa.string()),        # galleryURL used (image) or "" (specifics)
    pa.field("specifics_src", pa.string()),        # "ebay" | "inferred" | "none"
])

# Target shard size: 100,000 vectors per file (~200MB at float32 512-dim)
# Use Snappy compression (fast decompression, good ratio for float data)
SHARD_SIZE   = 100_000
COMPRESSION  = "snappy"
```

---

### 16.5 Vector Store API

```python
# src/embeddings/vector_store.py

import boto3
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import io, os
from dataclasses import dataclass
from datetime import datetime, timezone

@dataclass
class VectorRecord:
    os_id:        str
    qdrant_id:    str
    index_name:   str
    index_type:   str
    vector:       list[float]
    vector_type:  str           # "image" | "specifics"
    model_id:     str
    params_hash:  str
    source_url:   str = ""
    specifics_src:str = "ebay"


class S3VectorStore:
    """
    Primary durable store for all generated vectors.
    Write vectors here BEFORE loading into Qdrant.
    """

    def __init__(self, bucket: str, prefix: str = "vectors"):
        self.bucket  = bucket
        self.prefix  = prefix.rstrip("/")
        self.s3      = boto3.client("s3")

    # ── Key construction ──────────────────────────────────────────

    def key_prefix(
        self,
        vector_type: str,
        model_id:    str,
        params_hash: str,
        index_type:  str,
        partition:   str,
    ) -> str:
        """
        Build the S3 key prefix for a given combination of attributes.
        All components are lowercased and spaces replaced with hyphens.
        """
        return "/".join([
            self.prefix,
            vector_type.lower(),
            model_id.lower().replace(" ", "-"),
            params_hash.lower(),
            index_type.lower().replace(" ", "-"),
            partition.lower(),
        ])

    def shard_key(self, prefix: str, shard_num: int) -> str:
        return f"{prefix}/{shard_num:04d}.parquet"

    @staticmethod
    def partition_for_index(index_name: str, index_type: str) -> str:
        """
        Derive the partition string from the index name and type.
        eBay dated indices partition by full date (YYYY-MM-DD).
        Non-eBay indices use "all" (they are not date-partitioned).
        """
        if index_type == "ebay-dated":
            return index_name   # e.g. "2024-01-15"
        return "all"

    # ── Write ─────────────────────────────────────────────────────

    def write_shard(self, records: list[VectorRecord], shard_num: int) -> str:
        """
        Serialise a list of VectorRecords to a Parquet file and upload to S3.
        Returns the S3 key of the written object.
        Raises on S3 write failure — do NOT proceed to Qdrant if this raises.
        """
        if not records:
            raise ValueError("Cannot write empty shard")

        # All records in a shard must share the same metadata (enforced by caller)
        r0 = records[0]
        prefix = self.key_prefix(
            r0.vector_type, r0.model_id, r0.params_hash,
            r0.index_type,  self.partition_for_index(r0.index_name, r0.index_type),
        )
        key = self.shard_key(prefix, shard_num)

        table = pa.table({
            "os_id":         [r.os_id         for r in records],
            "qdrant_id":     [r.qdrant_id      for r in records],
            "index_name":    [r.index_name     for r in records],
            "index_type":    [r.index_type     for r in records],
            "vector":        [r.vector         for r in records],
            "vector_type":   [r.vector_type    for r in records],
            "model_id":      [r.model_id       for r in records],
            "params_hash":   [r.params_hash    for r in records],
            "created_at":    [datetime.now(timezone.utc) for _ in records],
            "source_url":    [r.source_url     for r in records],
            "specifics_src": [r.specifics_src  for r in records],
        })

        buf = io.BytesIO()
        pq.write_table(table, buf, compression="snappy")
        buf.seek(0)

        self.s3.put_object(Bucket=self.bucket, Key=key, Body=buf.getvalue())
        return key

    # ── Read / Repopulate ─────────────────────────────────────────

    def list_shards(
        self,
        vector_type:  str,
        model_id:     str,
        params_hash:  str,
        index_type:   str | None = None,
        partition:    str | None = None,
    ) -> list[str]:
        """List all shard keys matching the given attributes."""
        prefix_parts = [self.prefix, vector_type, model_id, params_hash]
        if index_type:
            prefix_parts.append(index_type)
            if partition:
                prefix_parts.append(partition)
        prefix = "/".join(prefix_parts) + "/"

        keys = []
        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith(".parquet"):
                    keys.append(obj["Key"])
        return sorted(keys)

    def read_shard(self, key: str) -> pa.Table:
        """Download and deserialise a single shard parquet file."""
        obj = self.s3.get_object(Bucket=self.bucket, Key=key)
        return pq.read_table(io.BytesIO(obj["Body"].read()))

    def iter_vectors(
        self,
        vector_type:  str,
        model_id:     str,
        params_hash:  str,
        index_type:   str | None = None,
        partition:    str | None = None,
        columns:      list[str] | None = None,
    ):
        """
        Iterate over VectorRecords from S3, shard by shard.
        Used by the repopulation pipeline — yields batches of rows.
        Specify columns=["os_id","qdrant_id","vector"] for fast Qdrant reload
        without reading provenance metadata columns.
        """
        keys = self.list_shards(vector_type, model_id, params_hash, index_type, partition)
        for key in keys:
            table = pq.read_table(
                io.BytesIO(self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()),
                columns=columns,
            )
            yield table

    # ── Checkpoint ────────────────────────────────────────────────

    def save_checkpoint(self, job_id: str, state: dict) -> None:
        """Persist ingestion progress so interrupted jobs resume from last batch."""
        import json
        key = f"{self.prefix.replace('vectors', 'checkpoints')}/{job_id}.json"
        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=json.dumps(state).encode(),
        )

    def load_checkpoint(self, job_id: str) -> dict | None:
        """Load checkpoint state. Returns None if no checkpoint exists."""
        import json
        key = f"{self.prefix.replace('vectors', 'checkpoints')}/{job_id}.json"
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=key)
            return json.loads(obj["Body"].read())
        except self.s3.exceptions.NoSuchKey:
            return None
```

---

### 16.6 Pipeline Write Order (Mandatory)

The pipeline always writes to S3 **before** upserting to Qdrant. This order is enforced by the pipeline and must not be changed:

```python
# src/ingestion/pipeline.py — mandatory write order

async def process_and_store(
    records:      list[VectorRecord],
    qdrant_client: QdrantClient,
    vector_store:  S3VectorStore,
    shard_num:    int,
) -> None:
    """
    Step 1: Write to S3 (durable store — must succeed before Qdrant)
    Step 2: Load into Qdrant (queryable index — rebuilt from S3 if lost)
    Raises immediately if S3 write fails. Never upserts to Qdrant without S3 confirmation.
    """
    # ── Step 1: S3 (always first) ─────────────────────────────────
    s3_key = vector_store.write_shard(records, shard_num)
    # If write_shard raises, execution stops here. Qdrant upsert is never reached.

    # ── Step 2: Qdrant (only after S3 confirmed) ──────────────────
    points = [
        PointStruct(
            id=r.qdrant_id,
            vector={r.vector_type: r.vector},
            payload={"os_id": r.os_id, ...},
        )
        for r in records
    ]
    qdrant_client.upsert(
        collection_name="cards",
        points=points,
        wait=False,   # async for backfill throughput
    )
```

---

### 16.7 Repopulation Procedure

Use this procedure whenever Qdrant needs to be rebuilt (node loss, collection reset, parameter change, or new cluster provisioning):

```bash
# 1. Verify S3 coverage — list all shards for the target model/type
python src/embeddings/vector_store.py list   --vector-type image   --model clip-vit-l-14   --params v1-fp16-224px

# 2. Rebuild Qdrant from S3 (no re-embedding, no OpenSearch access required)
python src/embeddings/vector_store.py repopulate   --s3-bucket $S3_VECTOR_BUCKET   --s3-prefix $S3_VECTOR_PREFIX   --vector-type image   --model clip-vit-l-14   --params v1-fp16-224px   --qdrant-host $QDRANT_HOST   --qdrant-collection cards   [--index-type ebay-dated]   # optional: scope to one index type
  [--partition 2024-01-15]    # optional: scope to one date partition

# 3. Verify Qdrant point count matches S3 record count
python src/embeddings/vector_store.py verify   --vector-type image   --model clip-vit-l-14   --params v1-fp16-224px
```

```python
# src/embeddings/vector_store.py — repopulate command implementation

def repopulate_qdrant(
    vector_store:   S3VectorStore,
    qdrant_client:  QdrantClient,
    vector_type:    str,
    model_id:       str,
    params_hash:    str,
    collection:     str = "cards",
    index_type:     str | None = None,
    partition:      str | None = None,
    batch_size:     int = 1_000,
) -> int:
    """
    Rebuild Qdrant from S3 vectors. Returns total points loaded.
    Safe to re-run — upsert is idempotent on qdrant_id.
    """
    total = 0
    for table in vector_store.iter_vectors(
        vector_type, model_id, params_hash, index_type, partition,
        columns=["os_id", "qdrant_id", "vector"],
    ):
        for batch_start in range(0, len(table), batch_size):
            batch = table.slice(batch_start, batch_size)
            points = [
                PointStruct(
                    id=batch["qdrant_id"][i].as_py(),
                    vector={vector_type: batch["vector"][i].as_py()},
                    payload={"os_id": batch["os_id"][i].as_py()},
                )
                for i in range(len(batch))
            ]
            qdrant_client.upsert(
                collection_name=collection,
                points=points,
                wait=False,
            )
            total += len(points)

    return total
```

---

### 16.8 Model Versioning and Future Embeddings

When a new embedding model or parameter set is introduced, create a new key prefix. The old vectors remain in S3:

```
vectors/image/clip-vit-l-14/v1-fp16-224px/...   ← original, intact
vectors/image/clip-vit-b-32/v1-fp32-224px/...   ← older model, kept for reference
vectors/image/clip-vit-h-14/v1-fp16-224px/...   ← future upgrade, new prefix
```

This means:
- Old Qdrant collections can be rebuilt from old S3 vectors if a rollback is needed
- New vectors can be backfilled into S3 and loaded into a new Qdrant named vector field in parallel (zero downtime model upgrade — see Section 13.5)
- Storage cost grows with each model version but is cheap relative to compute: 50M × 512-dim float32 ≈ 100GB ≈ ~$2.30/month on S3 Standard

---

### 16.9 S3 Bucket Configuration Recommendations

```
Bucket settings:
  Versioning:          Enabled (protects against accidental overwrites)
  Lifecycle policy:    Move to S3-IA after 90 days (infrequent access, ~40% cheaper)
                       Move to S3-Glacier after 365 days for old model versions
  Server-side encryption: SSE-S3 or SSE-KMS
  Block public access: All public access blocked
  Access:              IAM role for EC2 batch workers; read-only role for application

Estimated storage costs (us-west-1, S3 Standard):
  50M items × 2 vector types × 512-dim float32:  ~200GB  → ~$4.60/month
  200M items × 2 vector types:                   ~800GB  → ~$18.40/month
  Per additional model version (50M items):       ~200GB  → ~$4.60/month
```

---

---

## 17. References

- Malkov, Y.A. and Yashunin, D.A. (2020) 'Efficient and robust approximate nearest neighbor search using HNSW', *IEEE TPAMI*, 42(4), pp. 824–836.
- Cormack, G.V. et al. (2009) 'Reciprocal rank fusion outperforms Condorcet', *ACM SIGIR*, pp. 758–759.
- Cherti, M. et al. (2023) 'Reproducible scaling laws for contrastive language-image learning', *CVPR*, pp. 2818–2829.
- Cohen, J. (1988) *Statistical Power Analysis for the Behavioral Sciences*. 2nd edn. Lawrence Erlbaum.
- Wilcoxon, F. (1945) 'Individual comparisons by ranking methods', *Biometrics Bulletin*, 1(6), pp. 80–83.
- Hochberg, Y. (1988) 'A sharper Bonferroni procedure', *Biometrika*, 75(4), pp. 800–802.
- Qdrant Documentation: https://qdrant.tech/documentation/
- OpenCLIP: https://github.com/mlfoundations/open_clip
- sentence-transformers/all-MiniLM-L6-v2: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

"""Main experiment orchestrator.

Runs the same query set against both OpenSearch KNN (control) and
Qdrant ANN (treatment) in parallel, collecting recall, latency,
and cost metrics for statistical comparison.
"""

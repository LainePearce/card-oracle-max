"""OpenSearch KNN search queries — control system for experiment.

Queries the extant AOS cluster using imageVector KNN.
post_filter is applied AFTER KNN candidate selection — this intentionally
causes recall degradation under selective filters (documented weakness).
"""

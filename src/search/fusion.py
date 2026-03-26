"""RRF score fusion and catalogue boost policy.

Merges ranked lists from Qdrant prefetch arms using Reciprocal Rank Fusion.
Applies type-specific boosts to catalogue cards based on specifics similarity
and provenance (eBay-provided vs inferred).
"""

BOOST_RULES = {
    "sold":                   0.00,
    "catalogue_image":        0.12,
    "catalogue_no_img_high":  0.15,
    "catalogue_no_img_mid":   0.05,
    "catalogue_no_img_low":  -0.10,
}

INFERRED_SPECIFICS_BOOST_MULTIPLIER = 0.5

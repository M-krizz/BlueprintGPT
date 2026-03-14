"""learned.data – Data pipeline for the LayoutTransformer.

Modules
-------
tokenizer_layout : Discretize / encode room geometries into integer tokens.
build_coco       : Convert annotation JSONs → COCO + FloorPlanSample JSON.
build_sequences  : Build training-ready token sequences from parsed plans.
"""

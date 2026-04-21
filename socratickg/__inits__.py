"""SocraticKG: Knowledge Graph Construction via QA-Driven Fact Extraction.

Official implementation of the ACL 2026 Findings paper.

This package builds knowledge graphs from unstructured text in three stages:

    1. QA generation    - Decompose documents into 5W1H-guided QA pairs.
    2. Triple extraction - Map each QA pair to atomic (e1, r, e2) triples.
    3. Canonicalization - Unify entities and relations into a cohesive graph.

Typical usage:

    from socratickg.extraction import run_extraction
    from socratickg.canonicalization import run_canonicalization

Or run the full pipeline from the command line:

    python -m socratickg.run --dataset <hf_dataset_name>

Reference:
    Choi, S., Jeon, W., Yang, K., & Kim, T. (2026).
    SocraticKG: Knowledge Graph Construction via QA-Driven Fact Extraction.
    In Findings of the Association for Computational Linguistics: ACL 2026.
"""

__author__ = "Sanghyeok Choi, Woosang Jeon, Kyuseok Yang, Taehyeong Kim"

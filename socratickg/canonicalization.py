"""Step 2: Entity and relation canonicalization. (From. KGGen: Extracting Knowledge Graphs from Plain Text with Language Models)

Pipeline:
  1. Embed all unique entities (and relations) with a sentence-transformer.
  2. Partition via K-means into clusters of at most CLUSTER_SIZE items.
  3. Inside each cluster, rank candidates by hybrid dense (cosine) + sparse (BM25) score.
  4. Use an LLM to map near-duplicates to a single canonical alias.
  5. Rewrite the raw triples with canonicalized entities/relations and deduplicate.
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
from tqdm import tqdm

from . import config
from .llm_client import call_model, parse_llm_json
from .prompts import RESOLUTION_PROMPT


_embedder: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    """Lazily load the sentence embedding model."""
    global _embedder
    if _embedder is None:
        print(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        _embedder = SentenceTransformer(config.EMBEDDING_MODEL)
    return _embedder


def _normalize(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize scores to [0, 1], safe against constant arrays."""
    return (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)


def _resolve_with_llm(target: str, candidates: list[str], item_type: str) -> dict:
    """Ask the LLM to identify duplicates and a canonical alias for `target`."""
    prompt = f"""{RESOLUTION_PROMPT.format(item_type=item_type)}

Target Item: "{target}"
Candidate List: {json.dumps(candidates)}

Please output the result in the following JSON format:
{{
    "duplicates": ["dup1", "dup2"],
    "alias": "canonical_name"
}}"""
    res_text, _ = call_model(prompt)
    res_json = parse_llm_json(res_text)
    if isinstance(res_json, list) and res_json:
        res_json = res_json[0]
    if not isinstance(res_json, dict):
        return {"duplicates": [], "alias": target}

    alias = str(res_json.get("alias", "")).strip() or target
    duplicates = res_json.get("duplicates", [])
    if not isinstance(duplicates, list):
        duplicates = []
    return {"duplicates": duplicates, "alias": alias}


def _resolve_items(items: list[str], item_type: str) -> dict[str, str]:
    """Build a {raw -> canonical} map for all unique values in `items`."""
    items = [x for x in items if x and str(x).strip()]
    if not items:
        return {}

    unique_items = list(set(items))

    # Embed + cluster
    try:
        embedder = get_embedder()
        embeddings = embedder.encode(unique_items, batch_size=32, convert_to_tensor=True)
        embeddings_np = embeddings.cpu().numpy()

        n_clusters = max(1, len(unique_items) // config.CLUSTER_SIZE)
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings_np)
    except Exception as e:
        # Too few items / numerical issues: identity mapping
        print(f"[resolve_items:{item_type}] clustering failed ({e}); using identity map.")
        return {x: x for x in unique_items}

    clusters: dict[int, list[str]] = {}
    for i, label in enumerate(cluster_labels):
        clusters.setdefault(label, []).append(unique_items[i])

    resolved_map: dict[str, str] = {}

    for cluster_items in clusters.values():
        tokenized = [t.split() for t in cluster_items]
        bm25 = BM25Okapi(tokenized)
        cluster_indices = [unique_items.index(t) for t in cluster_items]
        cluster_embs = embeddings[cluster_indices]

        remaining = set(cluster_items)
        while remaining:
            target = next(iter(remaining))
            t_idx = cluster_items.index(target)

            cos_scores = util.cos_sim(cluster_embs[t_idx], cluster_embs)[0].cpu().numpy()
            bm25_scores = np.array(bm25.get_scores(target.split()))
            hybrid = 0.5 * _normalize(cos_scores) + 0.5 * _normalize(bm25_scores)

            top_k = np.argsort(hybrid)[::-1][:config.TOP_K_CANDIDATES]
            candidates = [cluster_items[i] for i in top_k if cluster_items[i] in remaining]

            if len(candidates) <= 1:
                resolved_map[target] = target
                remaining.remove(target)
                continue

            try:
                result = _resolve_with_llm(target, candidates, item_type)
                canonical = result["alias"]
                group = set(result["duplicates"]) | {target}
                valid = [x for x in group if x in remaining]
                for item in valid:
                    resolved_map[item] = canonical
                    remaining.remove(item)
            except Exception:
                resolved_map[target] = target
                remaining.remove(target)

    return resolved_map


def _apply_mapping(raw_triples: list[dict],
                   entity_map: dict[str, str],
                   relation_map: dict[str, str]) -> list[dict]:
    """Rewrite triples with canonical forms, drop empties and self-loops, dedupe."""
    final_triples: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    for t in raw_triples:
        e1_raw, e2_raw, r_raw = t["entity1"], t["entity2"], t["relation"]

        e1 = entity_map.get(e1_raw, e1_raw) or e1_raw
        e2 = entity_map.get(e2_raw, e2_raw) or e2_raw
        r = relation_map.get(r_raw, r_raw) or r_raw

        if not (str(e1).strip() and str(e2).strip() and str(r).strip()):
            continue
        if e1 == e2:
            continue

        key = (e1, r, e2)
        if key in seen:
            continue
        seen.add(key)
        final_triples.append({"entity1": e1, "relation": r, "entity2": e2})

    return final_triples


def resolve_document(raw_triple_path: Path, final_triple_dir: Path) -> str:
    """Canonicalize a single document's raw triples."""
    doc_id = raw_triple_path.name.split("_")[0]
    out_path = final_triple_dir / f"{doc_id}triples.json"

    if out_path.exists():
        return f"[{doc_id}] skipped (already exists)"

    try:
        with open(raw_triple_path, "r", encoding="utf-8") as f:
            raw_triples = json.load(f)
    except Exception as e:
        return f"[{doc_id}] load error: {e}"

    if not raw_triples:
        return f"[{doc_id}] empty"

    entities = [t["entity1"] for t in raw_triples] + [t["entity2"] for t in raw_triples]
    entity_map = _resolve_items(entities, item_type="entity")
    relation_map = _resolve_items([t["relation"] for t in raw_triples], item_type="predicate")

    final_triples = _apply_mapping(raw_triples, entity_map, relation_map)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_triples, f, indent=2, ensure_ascii=False)

    return f"[{doc_id}] resolved: {len(raw_triples)} -> {len(final_triples)}"


def run_canonicalization(
    raw_triple_dir: Path,
    final_triple_dir: Path,
    max_workers: int = 3,
) -> None:
    """Canonicalize every `*_raw_triples.json` file in `raw_triple_dir`."""
    files = sorted(raw_triple_dir.glob("*_raw_triples.json"))
    if not files:
        print(f"No raw triple files found in {raw_triple_dir}")
        return

    print(f"Starting canonicalization on {len(files)} files (max_workers={max_workers})...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(resolve_document, f, final_triple_dir): f for f in files
        }
        for future in tqdm(as_completed(futures), total=len(files)):
            try:
                future.result()
            except Exception as e:
                print(f"worker error: {e}")

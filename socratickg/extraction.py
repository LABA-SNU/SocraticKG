"""Step 1: 5W1H-guided QA generation and triple extraction.

For each document:
  1. Generate context-independent QA pairs from the full document.
  2. Extract atomic (entity1, relation, entity2) triples from each QA pair.

Supports parallel processing across documents via ThreadPoolExecutor.
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Mapping

from tqdm import tqdm

from llm_client import call_model, parse_llm_json
from prompts import WHOLE_DOC_QA_PROMPT, EXTRACTION_PROMPT


def _generate_qa_pairs(content: str, qa_out_path: Path) -> tuple[list[dict], int]:
    """Generate 5W1H-guided QA pairs for a single document.

    Reads from cache if `qa_out_path` already exists; otherwise calls the LLM
    and writes the result to disk.
    """
    if qa_out_path.exists():
        with open(qa_out_path, "r", encoding="utf-8") as f:
            return json.load(f), 0

    prompt = WHOLE_DOC_QA_PROMPT.format(document_text=content)
    res_text, usage = call_model(prompt)
    qa_pairs = parse_llm_json(res_text) or []

    # Some models prefix questions with bracketed tags (e.g. "[WHO] ...").
    # Strip them for clean downstream use.
    for item in qa_pairs:
        if "question" in item:
            item["question"] = re.sub(r"^\[.*?\]\s*", "", item["question"])

    with open(qa_out_path, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)

    return qa_pairs, usage.get("total_tokens", 0)


def _extract_triples_from_qa(
    qa_pairs: list[dict], source_id: int
) -> tuple[list[dict], int]:
    """Extract atomic triples from a list of QA pairs."""
    raw_triples: list[dict] = []
    total_tokens = 0

    for qa in qa_pairs:
        question = qa.get("question")
        answer = qa.get("answer")
        if not question or not answer:
            continue

        prompt = EXTRACTION_PROMPT.format(question=question, answer=answer)
        res_text, usage = call_model(prompt)
        total_tokens += usage.get("total_tokens", 0)

        for cand in parse_llm_json(res_text):
            if not all(k in cand and str(cand[k]).strip() for k in ("entity1", "relation", "entity2")):
                continue
            cand["entity1"] = str(cand["entity1"]).strip()
            cand["entity2"] = str(cand["entity2"]).strip()
            cand["relation"] = str(cand["relation"]).strip()
            if cand["entity1"] and cand["entity2"] and cand["relation"]:
                cand["source_id"] = source_id
                raw_triples.append(cand)

    return raw_triples, total_tokens


def process_document(
    idx: int,
    row: Mapping,
    qa_dir: Path,
    raw_triple_dir: Path,
    usage_dir: Path,
    text_field: str = "content",
) -> str:
    """Run QA generation + triple extraction for a single document.

    Skips the document if the final raw-triples file already exists (resumable).
    """
    content = row[text_field]
    qa_path = qa_dir / f"{idx}qa.json"
    triple_path = raw_triple_dir / f"{idx}_raw_triples.json"
    usage_path = usage_dir / f"{idx}_usage.json"

    if triple_path.exists():
        return f"[{idx}] skipped (already done)"

    token_stats = {"total_tokens": 0, "details": {"qa_gen": 0, "extraction": 0}}

    # Phase 1: QA generation
    try:
        qa_pairs, qa_tokens = _generate_qa_pairs(content, qa_path)
        token_stats["details"]["qa_gen"] = qa_tokens
    except Exception as e:
        return f"[{idx}] QA generation error: {e}"

    # Phase 2: Triple extraction
    raw_triples, ex_tokens = _extract_triples_from_qa(qa_pairs, idx)
    token_stats["details"]["extraction"] = ex_tokens
    token_stats["total_tokens"] = sum(token_stats["details"].values())

    with open(triple_path, "w", encoding="utf-8") as f:
        json.dump(raw_triples, f, indent=2, ensure_ascii=False)
    with open(usage_path, "w", encoding="utf-8") as f:
        json.dump(token_stats, f, indent=2)

    return f"[{idx}] QA: {len(qa_pairs)} -> Triples: {len(raw_triples)}"


def run_extraction(
    dataset: Iterable,
    qa_dir: Path,
    raw_triple_dir: Path,
    usage_dir: Path,
    max_workers: int = 3,
    text_field: str = "content",
) -> None:
    """Run QA generation + triple extraction across a dataset in parallel."""
    rows = list(dataset)
    print(f"Starting QA + triple extraction on {len(rows)} documents "
          f"(max_workers={max_workers})...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_document, i, row, qa_dir, raw_triple_dir, usage_dir, text_field
            ): i
            for i, row in enumerate(rows, 1)
        }
        for future in tqdm(as_completed(futures), total=len(rows)):
            try:
                future.result()
            except Exception as e:
                idx = futures[future]
                print(f"[{idx}] worker error: {e}")

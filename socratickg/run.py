"""CLI entry point for the SocraticKG pipeline.

Runs the full pipeline on a HuggingFace dataset:
    1. 5W1H-guided QA generation + triple extraction
    2. Entity and relation canonicalization

Example:
    python run.py --dataset kyssen/kg-gen-evaluation-essays --split train
    python run.py --dataset kyssen/kg-gen-evaluation-essays --steps extract
    python run.py --steps canonicalize --output-dir outputs/my_run
"""

import argparse
from pathlib import Path

from datasets import load_dataset

from . import config
from .extraction import run_extraction
from .canonicalization import run_canonicalization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the SocraticKG knowledge graph construction pipeline.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="kyssen/kg-gen-evaluation-essays",
        help="HuggingFace dataset name to process.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (default: train).",
    )
    parser.add_argument(
        "--text-field",
        type=str,
        default="content",
        help="Name of the column containing the document text (default: content).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.DEFAULT_OUTPUT_DIR,
        help=f"Base output directory (default: {config.DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--steps",
        type=str,
        nargs="+",
        choices=["extract", "canonicalize", "all"],
        default=["all"],
        help="Which pipeline steps to run (default: all).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=3,
        help="Number of parallel workers per step (default: 3).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dirs = config.build_output_dirs(args.output_dir)

    run_all = "all" in args.steps
    do_extract = run_all or "extract" in args.steps
    do_canon = run_all or "canonicalize" in args.steps

    if do_extract:
        print(f"Loading dataset: {args.dataset} [{args.split}]")
        dataset = load_dataset(args.dataset)[args.split]
        run_extraction(
            dataset=dataset,
            qa_dir=dirs["qa"],
            raw_triple_dir=dirs["raw_triples"],
            usage_dir=dirs["usage"],
            max_workers=args.max_workers,
            text_field=args.text_field,
        )

    if do_canon:
        run_canonicalization(
            raw_triple_dir=dirs["raw_triples"],
            final_triple_dir=dirs["final_triples"],
            max_workers=args.max_workers,
        )

    print(f"Done. Outputs at: {args.output_dir}")


if __name__ == "__main__":
    main()

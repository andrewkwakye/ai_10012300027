# CS4241 - Introduction to Artificial Intelligence
# Author: Andrew Kofi Kwakye
# Index: 10012300027
"""
Run the adversarial / baseline evaluation suite and dump to CSV.

Default behaviour:
    Runs DEFAULT_CASES in src/evaluator.py with 2 runs each, both with and
    without retrieval, then writes logs/evaluation_results.csv.

You can also inspect a single query with full debug:
    python scripts/run_evaluation.py --query "Who won Ablekuma in 2020?" --alpha 0.6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluator import run_suite  # noqa: E402
from src.pipeline import RAGPipeline  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", help="Ad-hoc query to debug (skips full suite)")
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--no-retrieval", action="store_true")
    ap.add_argument("--prompt", default=None)
    args = ap.parse_args()

    pipe = RAGPipeline()

    if args.query:
        kwargs = {"use_retrieval": not args.no_retrieval}
        if args.alpha is not None:
            kwargs["alpha"] = args.alpha
        if args.top_k is not None:
            kwargs["top_k"] = args.top_k
        if args.prompt is not None:
            kwargs["prompt_name"] = args.prompt
        r = pipe.ask(args.query, **kwargs)

        print("\n=== RETRIEVED ===")
        for i, ret in enumerate(r.retrieved, 1):
            print(
                f"#{i:02d}  score={ret.score:.3f}  dense={ret.dense_score:.3f}  "
                f"bm25={ret.bm25_score:.3f}  src={ret.meta.get('source')}"
            )
            print(f"     {ret.meta.get('text','')[:160].replace(chr(10),' ')}...")
        print("\n=== PROMPT (system) ===")
        print(r.prompt_system)
        print("\n=== PROMPT (user) ===")
        print(r.prompt_user)
        print("\n=== ANSWER ===")
        print(r.answer)
        if r.low_confidence:
            print(f"\n!! low-confidence: {r.note}")
        return

    rows = run_suite(pipe)
    print(f"\nWrote {len(rows)} rows. Summary:")
    rag = [r for r in rows if r["system"] == "RAG"]
    base = [r for r in rows if r["system"] == "LLM-only"]
    print(f"  RAG      halluc rate: {sum(r['hallucination'] for r in rag)}/{len(rag)}")
    print(f"  LLM-only halluc rate: {sum(r['hallucination'] for r in base)}/{len(base)}")


if __name__ == "__main__":
    main()

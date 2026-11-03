"""Retrieval benchmark: recall@k and MRR on labelled EN/DE questions.

Every question in `data/eval/questions.jsonl` is labelled with its language
and the chunk id(s) that answer it. Each corpus section has one English and
one German question, so half of the set is same-language and half is
cross-lingual (German question -> English document or the other way round).
Results are reported overall and split by direction, which is where lexical
retrievers fall apart.

    python -m src.evaluation --retrievers bm25 hashing hybrid-hashing
    python -m src.evaluation --retrievers multilingual hybrid-multilingual --json results.json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, Sequence

from src.embeddings import HashingEmbedder, VectorIndex
from src.ingest import Chunk, load_corpus
from src.retrieval import BM25Index, HybridRetriever

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "data" / "docs"
QUESTIONS = ROOT / "data" / "eval" / "questions.jsonl"


class Retriever(Protocol):
    def add(self, ids: Sequence[str], texts: Sequence[str]) -> None: ...
    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]: ...


@dataclass(frozen=True)
class Question:
    id: str
    lang: str
    question: str
    relevant: tuple[str, ...]


def load_questions(path: str | Path = QUESTIONS) -> list[Question]:
    questions = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            q = json.loads(line)
            questions.append(Question(q["id"], q["lang"], q["question"], tuple(q["relevant"])))
    return questions


def recall_at_k(ranked: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Share of the relevant chunks that appear in the top k."""
    return len(set(ranked[:k]) & set(relevant)) / len(relevant)


def reciprocal_rank(ranked: Sequence[str], relevant: Sequence[str]) -> float:
    """1 / rank of the first relevant chunk, 0 if none was retrieved."""
    for rank, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def direction(question: Question, doc_lang: dict[str, str]) -> str:
    """e.g. "de->en"; uses the language of the first relevant chunk."""
    return f"{question.lang}->{doc_lang[question.relevant[0]]}"


def evaluate(retriever: Retriever, questions: Sequence[Question], chunks: Sequence[Chunk],
             ks: Sequence[int] = (1, 3, 5), depth: int = 10) -> dict:
    """Index the chunks, run every question and aggregate the metrics.

    Returns {"groups": {name: {"n", "recall@k"..., "mrr"}}, "per_question": [...]}
    where the groups are overall, same-language, cross-lingual and each
    direction. MRR is computed over the top `depth` results.
    """
    retriever.add([c.id for c in chunks], [f"{c.title} {c.section} {c.text}" for c in chunks])
    doc_lang = {c.id: c.lang for c in chunks}
    rows = []
    for q in questions:
        ranked = [doc_id for doc_id, _ in retriever.search(q.question, top_k=max(depth, *ks))]
        d = direction(q, doc_lang)
        src, dst = d.split("->")
        row = {"id": q.id, "direction": d, "cross_lingual": src != dst, "top": ranked[:max(ks)],
               "rr": reciprocal_rank(ranked[:depth], q.relevant)}
        row.update({f"recall@{k}": recall_at_k(ranked, q.relevant, k) for k in ks})
        rows.append(row)

    def aggregate(subset: list[dict]) -> dict:
        n = len(subset)
        out = {"n": n}
        out.update({f"recall@{k}": sum(r[f"recall@{k}"] for r in subset) / n if n else 0.0 for k in ks})
        out["mrr"] = sum(r["rr"] for r in subset) / n if n else 0.0
        return out

    groups = {
        "all": aggregate(rows),
        "same-language": aggregate([r for r in rows if not r["cross_lingual"]]),
        "cross-lingual": aggregate([r for r in rows if r["cross_lingual"]]),
    }
    for d in sorted({r["direction"] for r in rows}):
        groups[d] = aggregate([r for r in rows if r["direction"] == d])
    return {"groups": groups, "per_question": rows}


def _multilingual_index() -> VectorIndex:
    from src.embeddings import MultilingualEmbedder  # needs sentence-transformers
    return VectorIndex(MultilingualEmbedder())


RETRIEVERS: dict[str, Callable[[], Retriever]] = {
    "bm25": BM25Index,
    "hashing": lambda: VectorIndex(HashingEmbedder()),
    "hybrid-hashing": lambda: HybridRetriever(VectorIndex(HashingEmbedder())),
    "multilingual": _multilingual_index,
    "hybrid-multilingual": lambda: HybridRetriever(_multilingual_index()),
}


def results_table(results: dict[str, dict], ks: Sequence[int] = (1, 3, 5),
                  groups: Sequence[str] = ("all", "same-language", "cross-lingual")) -> str:
    """Markdown table: one row per retriever and group."""
    header = ["retriever", "questions", "n"] + [f"recall@{k}" for k in ks] + ["MRR"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for name, res in results.items():
        for g in groups:
            m = res["groups"][g]
            cells = [name, g, str(m["n"])] + [f"{m[f'recall@{k}']:.3f}" for k in ks] + [f"{m['mrr']:.3f}"]
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--retrievers", nargs="+", default=["bm25", "hashing", "hybrid-hashing"],
                        choices=sorted(RETRIEVERS))
    parser.add_argument("--k", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--questions", default=str(QUESTIONS))
    parser.add_argument("--docs", default=str(DOCS))
    parser.add_argument("--json", help="also write full results (incl. per-question rankings) here")
    args = parser.parse_args(argv)

    chunks, questions = load_corpus(args.docs), load_questions(args.questions)
    results = {name: evaluate(RETRIEVERS[name](), questions, chunks, ks=args.k) for name in args.retrievers}
    print(results_table(results, ks=args.k))
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Groundedness and refusal evaluation of the full pipeline (retrieval + answer).

Two question sets:

- the 32 benchmark questions in `data/eval/questions.jsonl`, each labelled
  with the facts a correct answer must state. A fact is a list of accepted
  spellings in English and German (e.g. "2 euros", "2 Euro"), so an answer
  quoted from the document and one translated by an LLM are both matched.
- 16 questions in `data/eval/unanswerable.jsonl` (8 EN, 8 DE) that sound
  like they belong to one of the documents but are not answered by any of
  them (a gym sauna, e-books in the library). The right reply is the fixed
  "I don't know" sentence.

Per answer it checks:

- refused: the reply is the "I don't know" sentence (either language).
- correct: not refused, and every fact appears (word-bounded, case-insensitive).
- grounded: not refused, cites at least one retrieved chunk, cites no id
  that was not retrieved, and every number in the answer (digits or number
  words two to twelve, EN/DE) occurs in a cited chunk. Numbers are what a
  model most often gets wrong in policy answers (prices, hours, deadlines),
  and the check works across languages without a judge model. It does not
  catch a wrong claim without a number.

    python -m src.grounding --extractive --retrievers bm25 hybrid-hashing --min-overlap 0 0.5
    RAG_LLM_BASE_URL=http://localhost:8000/v1 RAG_LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct \\
        python -m src.grounding --retrievers hybrid-multilingual
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from src.evaluation import DOCS, QUESTIONS, RETRIEVERS
from src.generation import _CITATION, NO_ANSWER, Answer, ExtractiveAnswerer, OpenAICompatibleClient, RAGPipeline
from src.ingest import Chunk, load_corpus

UNANSWERABLE = QUESTIONS.parent / "unanswerable.jsonl"

_NUMBER_WORDS = {w: str(n) for n, words in enumerate([
    (), (), ("two", "zwei"), ("three", "drei"), ("four", "vier"), ("five", "fünf"), ("six", "sechs"),
    ("seven", "sieben"), ("eight", "acht"), ("nine", "neun"), ("ten", "zehn"), ("eleven", "elf"),
    ("twelve", "zwölf"),
]) for w in words}
_NUMBER = re.compile(r"\d+|\b(?:" + "|".join(_NUMBER_WORDS) + r")\b", re.I)


@dataclass(frozen=True)
class EvalQuestion:
    id: str
    lang: str
    question: str
    relevant: tuple[str, ...] = ()            # empty for unanswerable questions
    facts: tuple[tuple[str, ...], ...] = ()

    @property
    def answerable(self) -> bool:
        return bool(self.relevant)


def load_eval_questions(answerable: str | Path = QUESTIONS,
                        unanswerable: str | Path = UNANSWERABLE) -> list[EvalQuestion]:
    out = []
    for path in (answerable, unanswerable):
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                q = json.loads(line)
                out.append(EvalQuestion(q["id"], q["lang"], q["question"], tuple(q.get("relevant", ())),
                                        tuple(tuple(f) for f in q.get("facts", ()))))
    return out


def is_refusal(text: str) -> bool:
    return any(sentence in text for sentence in NO_ANSWER.values())


def has_fact(text: str, spellings: Sequence[str]) -> bool:
    return any(re.search(r"(?<!\w)" + re.escape(s) + r"(?!\w)", text, re.I) for s in spellings)


def numbers(text: str) -> set[str]:
    """Numbers in the text, without citation markers; number words become digits."""
    text = _CITATION.sub(" ", text)
    return {_NUMBER_WORDS.get(m.lower(), m.lstrip("0") or "0") for m in _NUMBER.findall(text)}


def check_answer(q: EvalQuestion, answer: Answer, chunks: dict[str, Chunk]) -> dict:
    refused = is_refusal(answer.text)
    invented = sorted(set(_CITATION.findall(answer.text)) - set(answer.context))
    cited_numbers = set().union(*(numbers(chunks[cid].text) for cid in answer.citations))
    unsupported = sorted(numbers(answer.text) - cited_numbers)
    grounded = not refused and bool(answer.citations) and not invented and not unsupported
    row = {"id": q.id, "lang": q.lang, "answerable": q.answerable, "mode": answer.mode, "refused": refused,
           "grounded": grounded, "citations": answer.citations, "invented_citations": invented,
           "unsupported_numbers": unsupported, "answer": answer.text}
    if q.answerable:
        doc_lang = chunks[q.relevant[0]].lang
        row["cross_lingual"] = q.lang != doc_lang
        row["correct"] = not refused and all(has_fact(answer.text, f) for f in q.facts)
        row["cites_relevant"] = bool(set(answer.citations) & set(q.relevant))
    return row


def _rate(rows: list[dict], key: str) -> float:
    return sum(r[key] for r in rows) / len(rows) if rows else 0.0


def evaluate(pipeline: RAGPipeline, questions: Sequence[EvalQuestion], chunks: Sequence[Chunk]) -> dict:
    """Ask every question and aggregate.

    answerable: correct, cites_relevant and false_refusal over all answerable
    questions, correct also split into same-language and cross-lingual.
    unanswerable: share refused. grounded: share of all non-refused answers
    (both sets) that are grounded.
    """
    by_id = {c.id: c for c in chunks}
    rows = [check_answer(q, pipeline.ask(q.question), by_id) for q in questions]
    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    answered = [r for r in rows if not r["refused"]]
    summary = {
        "answerable": len(ans),
        "correct": _rate(ans, "correct"),
        "correct_same_language": _rate([r for r in ans if not r["cross_lingual"]], "correct"),
        "correct_cross_lingual": _rate([r for r in ans if r["cross_lingual"]], "correct"),
        "cites_relevant": _rate(ans, "cites_relevant"),
        "false_refusal": _rate(ans, "refused"),
        "unanswerable": len(unans),
        "refused_unanswerable": _rate(unans, "refused"),
        "answers": len(answered),
        "grounded": _rate(answered, "grounded"),
        "modes": sorted({r["mode"] for r in rows}),
    }
    return {"summary": summary, "per_question": rows}


def results_table(results: dict[str, dict]) -> str:
    header = ["config", "correct", "same-lang", "cross-lingual", "cites relevant", "false refusals",
              "refused unanswerable", "grounded answers"]
    lines = ["| " + " | ".join(header) + " |", "|---|" + "---:|" * (len(header) - 1)]
    for name, res in results.items():
        s = res["summary"]
        cells = [name] + [f"{s[k]:.3f}" for k in ("correct", "correct_same_language", "correct_cross_lingual",
                                                  "cites_relevant", "false_refusal", "refused_unanswerable")]
        cells.append(f"{s['grounded']:.3f} ({s['answers']})")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--retrievers", nargs="+", default=["bm25", "hybrid-hashing"], choices=sorted(RETRIEVERS))
    parser.add_argument("--min-overlap", nargs="+", type=float, default=[0.0],
                        help="extractive abstention thresholds to compare (ignored when an LLM answers)")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--extractive", action="store_true", help="ignore RAG_LLM_* and answer extractively")
    parser.add_argument("--docs", default=str(DOCS))
    parser.add_argument("--questions", default=str(QUESTIONS))
    parser.add_argument("--unanswerable", default=str(UNANSWERABLE))
    parser.add_argument("--json", help="also write per-question results here")
    args = parser.parse_args(argv)

    chunks = load_corpus(args.docs)
    questions = load_eval_questions(args.questions, args.unanswerable)
    llm = None if args.extractive else OpenAICompatibleClient.from_env()
    results = {}
    for name in args.retrievers:
        retriever = RETRIEVERS[name]()
        retriever.add([c.id for c in chunks], [f"{c.title} {c.section} {c.text}" for c in chunks])
        if llm is not None:
            pipeline = RAGPipeline(retriever, chunks, llm, top_k=args.top_k)
            results[f"{name} + llm {llm.model}"] = evaluate(pipeline, questions, chunks)
            continue
        for t in args.min_overlap:
            pipeline = RAGPipeline(retriever, chunks, top_k=args.top_k, extractive=ExtractiveAnswerer(min_overlap=t))
            results[f"{name} + extractive, min_overlap={t:g}"] = evaluate(pipeline, questions, chunks)
    print(results_table(results))
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

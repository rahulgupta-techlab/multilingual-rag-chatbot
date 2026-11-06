"""Answer generation: retrieve chunks, then answer in the user's language with citations.

Two answerers share one interface:

- LLMAnswerer calls any OpenAI-compatible chat completions endpoint (vLLM,
  Ollama, OpenAI). The prompt holds the retrieved chunks with their ids and
  asks for an answer in the user's language that cites every chunk it uses
  as [chunk-id]. Citations to ids that were not in the context are dropped.
- ExtractiveAnswerer needs no model: it returns the retrieved sentences that
  share the most terms with the question, quoted in the document's language
  and cited. It is the fallback when no LLM is configured or the call fails.

    python -m src.generation "Wie viel kostet eine Fahrt ohne Abo?"
    RAG_LLM_BASE_URL=http://localhost:11434/v1 RAG_LLM_MODEL=qwen2.5:7b-instruct \\
        python -m src.generation "How long can I borrow a book?"
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol, Sequence

from src.ingest import Chunk
from src.language import detect_language
from src.retrieval import tokenize

NO_ANSWER = {
    "en": "I don't know. The documents do not contain an answer to this question.",
    "de": "Das weiß ich nicht. Die Dokumente enthalten keine Antwort auf diese Frage.",
}
_LANG_NAMES = {"en": "English", "de": "German"}
_CITATION = re.compile(r"\[([^\[\]\s]+#[^\[\]\s]+)\]")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


class Retriever(Protocol):
    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]: ...


class ChatClient(Protocol):
    def complete(self, messages: list[dict]) -> str: ...


@dataclass
class Answer:
    text: str
    citations: list[str]          # chunk ids, in order of first use
    lang: str                     # language of the question, "en" if unknown
    mode: str                     # "llm" | "extractive" | "no-context"
    context: list[str] = field(default_factory=list)   # retrieved chunk ids given to the answerer
    fallback_reason: str | None = None


def answer_language(question: str) -> str:
    lang = detect_language(question)
    return lang if lang in NO_ANSWER else "en"


def build_messages(question: str, chunks: Sequence[Chunk], lang: str) -> list[dict]:
    language = _LANG_NAMES[lang]
    system = (
        "You answer questions about policy documents using only the context passages below. "
        f"Always answer in {language}, even when a passage is in another language. "
        "After every statement, cite the passage it comes from by its id in square brackets, "
        "for example [bike_rental_en#prices]. Only cite ids that appear in the context. "
        f"If the context does not contain the answer, reply exactly: {NO_ANSWER[lang]}"
    )
    context = "\n\n".join(f"[{c.id}] {c.title} – {c.section}\n{c.text}" for c in chunks)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]


def extract_citations(text: str, allowed: Sequence[str]) -> list[str]:
    """[chunk-id] markers in the order they first appear, keeping only allowed ids."""
    allowed_set, seen = set(allowed), []
    for cid in _CITATION.findall(text):
        if cid in allowed_set and cid not in seen:
            seen.append(cid)
    return seen


def question_overlap(question: str, sentences: Sequence[str]) -> float:
    """Share of the question's distinct terms (BM25 tokenizer) that occur in the sentences."""
    terms = set(tokenize(question))
    if not terms:
        return 0.0
    found = {t for sent in sentences for t in tokenize(sent)}
    return len(terms & found) / len(terms)


class OpenAICompatibleClient:
    """Minimal chat completions client (stdlib only).

    base_url includes the API prefix, e.g. http://localhost:8000/v1 for vLLM,
    http://localhost:11434/v1 for Ollama or https://api.openai.com/v1.
    """

    def __init__(self, base_url: str, model: str, api_key: str | None = None, timeout: float = 30.0,
                 temperature: float = 0.0, max_tokens: int = 400):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.api_key, self.timeout = model, api_key, timeout
        self.temperature, self.max_tokens = temperature, max_tokens

    @classmethod
    def from_env(cls) -> "OpenAICompatibleClient | None":
        """RAG_LLM_BASE_URL, RAG_LLM_MODEL and optional RAG_LLM_API_KEY; None if not configured."""
        base_url = os.environ.get("RAG_LLM_BASE_URL")
        if not base_url:
            return None
        return cls(base_url, os.environ.get("RAG_LLM_MODEL", "default"), os.environ.get("RAG_LLM_API_KEY"))

    def complete(self, messages: list[dict]) -> str:
        body = json.dumps({"model": self.model, "messages": messages, "temperature": self.temperature,
                           "max_tokens": self.max_tokens}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        content = data["choices"][0]["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError("empty completion")
        return content.strip()


class ExtractiveAnswerer:
    """Quote the retrieved sentences that best match the question.

    A sentence scores the summed IDF of the question terms it contains (same
    tokenizer as BM25, IDF over the retrieved sentences), so a rare word like
    "password" counts for more than "ride", which every bike-rental sentence
    shares. The best sentence is always kept; up to max_sentences - 1 more from
    the same chunk are added if they score at least min_ratio of the best, so
    the answer stays inside one section. Ties go to the higher-ranked chunk,
    then the earlier sentence. If nothing overlaps, which
    is typical for cross-lingual questions, the first sentence of the top
    chunk is used, since retrieval already judged that chunk the most
    relevant. No translation happens here: the sentences stay in the
    document's language.

    min_overlap makes it abstain: if the picked sentences contain less than
    this share of the question's terms, it returns the "I don't know" answer
    instead of quoting. The default 0 never abstains; any value above 0 also
    abstains on questions with no overlap, i.e. most cross-lingual ones.
    """

    def __init__(self, max_sentences: int = 2, min_ratio: float = 0.5, min_overlap: float = 0.0):
        self.max_sentences, self.min_ratio, self.min_overlap = max_sentences, min_ratio, min_overlap

    def answer(self, question: str, chunks: Sequence[Chunk], lang: str) -> tuple[str, list[str]]:
        sentences = [(rank, i, chunk.id, sent, set(tokenize(sent)))
                     for rank, chunk in enumerate(chunks)
                     for i, sent in enumerate(s for s in _SENTENCE.split(chunk.text) if s)]
        if not sentences:
            return NO_ANSWER[lang], []
        n = len(sentences)
        idf = {t: math.log(1 + n / sum(t in toks for *_, toks in sentences)) for t in set(tokenize(question))
               if any(t in toks for *_, toks in sentences)}
        scored = sorted(((sum(idf.get(t, 0.0) for t in toks), rank, i, cid, sent)
                         for rank, i, cid, sent, toks in sentences), key=lambda c: (-c[0], c[1], c[2]))
        best = scored[0][0]
        if best > 0:
            top_cid = scored[0][3]
            picked = [c for c in scored if c[3] == top_cid and c[0] >= self.min_ratio * best][:self.max_sentences]
        else:
            picked = [min(scored, key=lambda c: (c[1], c[2]))]
        picked.sort(key=lambda c: (c[1], c[2]))
        if self.min_overlap > 0 and question_overlap(question, [c[4] for c in picked]) < self.min_overlap:
            return NO_ANSWER[lang], []
        cid = picked[0][3]
        return " ".join(c[4] for c in picked) + f" [{cid}]", [cid]


class RAGPipeline:
    def __init__(self, retriever: Retriever, chunks: Sequence[Chunk], llm: ChatClient | None = None,
                 top_k: int = 3, extractive: ExtractiveAnswerer | None = None):
        self.retriever = retriever
        self.chunks = {c.id: c for c in chunks}
        self.llm = llm
        self.top_k = top_k
        self.extractive = extractive or ExtractiveAnswerer()

    def ask(self, question: str, top_k: int | None = None) -> Answer:
        lang = answer_language(question)
        hits = [self.chunks[cid] for cid, _ in self.retriever.search(question, top_k=top_k or self.top_k)
                if cid in self.chunks]
        ids = [c.id for c in hits]
        if not hits:
            return Answer(NO_ANSWER[lang], [], lang, "no-context", ids)
        reason = None
        if self.llm is None:
            reason = "no LLM configured"
        else:
            try:
                text = self.llm.complete(build_messages(question, hits, lang))
                return Answer(text, extract_citations(text, ids), lang, "llm", ids)
            except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:  # URLError, timeouts, bad JSON
                reason = f"LLM call failed: {type(exc).__name__}: {exc}"
        text, citations = self.extractive.answer(question, hits, lang)
        return Answer(text, citations, lang, "extractive", ids, reason)


def main(argv: Sequence[str] | None = None) -> int:
    from src.evaluation import DOCS, RETRIEVERS
    from src.ingest import load_corpus

    parser = argparse.ArgumentParser(description="Answer a question from the documents, with citations.")
    parser.add_argument("question")
    parser.add_argument("--retriever", default="hybrid-hashing", choices=sorted(RETRIEVERS))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--docs", default=str(DOCS))
    parser.add_argument("--extractive", action="store_true", help="ignore RAG_LLM_* and answer extractively")
    args = parser.parse_args(argv)

    chunks = load_corpus(args.docs)
    retriever = RETRIEVERS[args.retriever]()
    retriever.add([c.id for c in chunks], [f"{c.title} {c.section} {c.text}" for c in chunks])
    llm = None if args.extractive else OpenAICompatibleClient.from_env()
    result = RAGPipeline(retriever, chunks, llm, top_k=args.top_k).ask(args.question)
    print(result.text)
    print(f"\nmode: {result.mode}" + (f" ({result.fallback_reason})" if result.fallback_reason else ""))
    print("sources: " + (", ".join(result.citations) or "none"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

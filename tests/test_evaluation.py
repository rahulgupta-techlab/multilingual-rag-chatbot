import json
from collections import Counter

from src.evaluation import (DOCS, Question, evaluate, load_questions, main, recall_at_k, reciprocal_rank,
                            results_table)
from src.ingest import load_corpus
from src.language import detect_language
from src.retrieval import BM25Index


class FixedRetriever:
    """Returns a scripted ranking per query, so the metrics can be checked by hand."""

    def __init__(self, rankings):
        self.rankings = rankings

    def add(self, ids, texts):
        pass

    def search(self, query, top_k=5):
        return [(doc_id, 1.0) for doc_id in self.rankings[query][:top_k]]


def test_recall_at_k_counts_share_of_relevant_in_top_k():
    ranked = ["a", "b", "c", "d"]
    assert recall_at_k(ranked, ["b"], 1) == 0.0
    assert recall_at_k(ranked, ["b"], 2) == 1.0
    assert recall_at_k(ranked, ["a", "d"], 3) == 0.5


def test_reciprocal_rank_uses_first_relevant_hit():
    assert reciprocal_rank(["a", "b", "c"], ["c", "b"]) == 0.5
    assert reciprocal_rank(["a", "b"], ["z"]) == 0.0


def test_question_set_covers_every_section_in_both_languages():
    questions = load_questions()
    chunk_ids = {c.id for c in load_corpus(DOCS)}
    assert len({q.id for q in questions}) == len(questions)
    for q in questions:
        assert set(q.relevant) <= chunk_ids, q.id
        assert detect_language(q.question) == q.lang, q.id
    per_section = Counter((q.relevant[0], q.lang) for q in questions)
    assert set(per_section) == {(cid, lang) for cid in chunk_ids for lang in ("en", "de")}


def test_evaluate_splits_same_language_and_cross_lingual():
    chunks = [c for c in load_corpus(DOCS) if c.id in {"bike_rental_en#prices", "stadtbibliothek_de#ausleihe"}]
    questions = [
        Question("1", "en", "price?", ("bike_rental_en#prices",)),         # en->en, rank 1
        Question("2", "de", "Preis?", ("bike_rental_en#prices",)),         # de->en, rank 2
        Question("3", "en", "borrow?", ("stadtbibliothek_de#ausleihe",)),  # en->de, not found
    ]
    retriever = FixedRetriever({
        "price?": ["bike_rental_en#prices", "stadtbibliothek_de#ausleihe"],
        "Preis?": ["stadtbibliothek_de#ausleihe", "bike_rental_en#prices"],
        "borrow?": ["bike_rental_en#prices"],
    })
    groups = evaluate(retriever, questions, chunks, ks=(1, 3))["groups"]
    assert groups["all"] == {"n": 3, "recall@1": 1 / 3, "recall@3": 2 / 3, "mrr": 0.5}
    assert groups["same-language"] == {"n": 1, "recall@1": 1.0, "recall@3": 1.0, "mrr": 1.0}
    assert groups["cross-lingual"] == {"n": 2, "recall@1": 0.0, "recall@3": 0.5, "mrr": 0.25}
    assert groups["de->en"]["mrr"] == 0.5 and groups["en->de"]["mrr"] == 0.0


def test_lexical_retrieval_is_much_weaker_across_languages():
    result = evaluate(BM25Index(), load_questions(), load_corpus(DOCS))
    same, cross = result["groups"]["same-language"], result["groups"]["cross-lingual"]
    assert same["n"] == cross["n"] == 16
    assert same["recall@5"] >= 0.75
    assert cross["recall@5"] < same["recall@5"] - 0.4


def test_cli_prints_table_and_writes_json(tmp_path, capsys):
    out = tmp_path / "results.json"
    assert main(["--retrievers", "bm25", "hybrid-hashing", "--json", str(out)]) == 0
    table = capsys.readouterr().out
    assert "| bm25 | cross-lingual | 16 |" in table
    data = json.loads(out.read_text(encoding="utf-8"))
    assert set(data) == {"bm25", "hybrid-hashing"}
    assert len(data["bm25"]["per_question"]) == 32
    assert results_table(data).splitlines()[0].startswith("| retriever |")

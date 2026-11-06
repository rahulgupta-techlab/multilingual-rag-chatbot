import json

from src.evaluation import DOCS
from src.generation import NO_ANSWER, Answer, ExtractiveAnswerer, RAGPipeline
from src.grounding import check_answer, evaluate, has_fact, is_refusal, load_eval_questions, main, numbers
from src.ingest import load_corpus
from src.language import detect_language
from src.retrieval import BM25Index

CHUNKS = {c.id: c for c in load_corpus(DOCS)}
QUESTIONS = {q.id: q for q in load_eval_questions()}


class FixedRetriever:
    def __init__(self, ids):
        self.ids = ids

    def search(self, query, top_k=5):
        return [(cid, 1.0) for cid in self.ids[:top_k]]


class ReplyPerQuestion:
    """Scripted LLM: looks up the reply by the question at the end of the prompt."""

    def __init__(self, replies, default):
        self.replies, self.default = replies, default

    def complete(self, messages):
        question = messages[-1]["content"].rsplit("Question: ", 1)[1]
        return self.replies.get(question, self.default)


def test_question_sets_are_labelled_and_answerable_ones_have_facts():
    questions = list(QUESTIONS.values())
    assert len(questions) == 48 and len({q.id for q in questions}) == 48
    unanswerable = [q for q in questions if not q.answerable]
    assert len(unanswerable) == 16 and sum(q.lang == "de" for q in unanswerable) == 8
    for q in questions:
        assert detect_language(q.question) == q.lang, q.id
        assert q.answerable == bool(q.facts), q.id


def test_every_fact_is_stated_in_the_relevant_section():
    # Gold facts must be checkable: at least one spelling of each occurs in the source text.
    for q in QUESTIONS.values():
        text = " ".join(CHUNKS[cid].text for cid in q.relevant)
        for fact in q.facts:
            assert has_fact(text, fact), (q.id, fact)


def test_fact_matching_is_word_bounded_and_case_insensitive():
    assert has_fact("Non-subscribers pay 2 Euro.", ["2 euro"])
    assert not has_fact("It costs 12 euro.", ["2 euro"])
    assert not has_fact("Open 24 hours.", ["2"])


def test_numbers_include_number_words_and_skip_citations():
    assert numbers("Vier Wochen, twice extended, 2 euros [x_en#a1]") == {"4", "2"}
    assert numbers("8:00 to 18:00") == {"8", "0", "18"}


def test_refusal_is_the_fixed_sentence_in_either_language():
    assert is_refusal(NO_ANSWER["de"]) and is_refusal("Sorry. " + NO_ANSWER["en"])
    assert not is_refusal("I don't know the exact price, but it is 2 euros.")


def test_grounded_needs_a_citation_and_only_numbers_from_cited_chunks():
    q = QUESTIONS["q03"]
    ctx = ["bike_rental_en#prices", "coworking_en#access"]

    def check(text, citations):
        return check_answer(q, Answer(text, citations, "en", "llm", ctx), CHUNKS)

    good = check("Non-subscribers pay 2 euros per 30 minutes. [bike_rental_en#prices]", ["bike_rental_en#prices"])
    assert good["grounded"] and good["correct"] and good["cites_relevant"]
    wrong_number = check("They pay 3 euros per 30 minutes. [bike_rental_en#prices]", ["bike_rental_en#prices"])
    assert not wrong_number["grounded"] and wrong_number["unsupported_numbers"] == ["3"]
    uncited = check("Non-subscribers pay 2 euros per 30 minutes.", [])
    assert not uncited["grounded"] and uncited["correct"]
    invented = check("2 euros [bike_rental_en#prices] [bike_rental_en#helmets]", ["bike_rental_en#prices"])
    assert not invented["grounded"] and invented["invented_citations"] == ["bike_rental_en#helmets"]
    wrong_chunk = check("Day-pass guests: 9:00 to 19:00. [coworking_en#access]", ["coworking_en#access"])
    assert wrong_chunk["grounded"] and not wrong_chunk["correct"] and not wrong_chunk["cites_relevant"]


def test_min_overlap_makes_the_extractive_answerer_abstain():
    chunks = [CHUNKS["fitnessstudio_de#kurse"]]
    question = "Gibt es im Fitnessstudio eine Sauna?"
    text, citations = ExtractiveAnswerer().answer(question, chunks, "de")
    assert citations == ["fitnessstudio_de#kurse"]
    text, citations = ExtractiveAnswerer(min_overlap=0.5).answer(question, chunks, "de")
    assert text == NO_ANSWER["de"] and citations == []
    text, citations = ExtractiveAnswerer(min_overlap=0.5).answer(
        "Sind Yoga Kurse im Beitrag enthalten?", chunks, "de")
    assert citations == ["fitnessstudio_de#kurse"] and "Yoga" in text


def test_evaluate_scores_a_scripted_llm():
    questions = [QUESTIONS[i] for i in ("q03", "q04", "u05", "u13")]
    llm = ReplyPerQuestion({
        QUESTIONS["q03"].question: "Non-subscribers pay 2 euros per 30 minutes. [bike_rental_en#prices]",
        QUESTIONS["q04"].question: NO_ANSWER["de"],                                  # false refusal
        QUESTIONS["u05"].question: NO_ANSWER["en"],                                  # right refusal
        QUESTIONS["u13"].question: "Ja, sonntags von 10 bis 14 Uhr. [stadtbibliothek_de#oeffnungszeiten]",
    }, default="unused")
    pipeline = RAGPipeline(FixedRetriever(["bike_rental_en#prices", "stadtbibliothek_de#oeffnungszeiten"]),
                           list(CHUNKS.values()), llm)
    s = evaluate(pipeline, questions, list(CHUNKS.values()))["summary"]
    assert s["answerable"] == 2 and s["correct"] == 0.5 and s["false_refusal"] == 0.5
    assert s["correct_same_language"] == 1.0 and s["correct_cross_lingual"] == 0.0
    assert s["unanswerable"] == 2 and s["refused_unanswerable"] == 0.5
    # the invented Sunday hours cite a real chunk, but 14 is not in it
    assert s["answers"] == 2 and s["grounded"] == 0.5 and s["modes"] == ["llm"]


def test_extractive_answers_are_grounded_but_answer_unanswerable_questions():
    bm25 = BM25Index()
    bm25.add(list(CHUNKS), [f"{c.title} {c.section} {c.text}" for c in CHUNKS.values()])
    s = evaluate(RAGPipeline(bm25, list(CHUNKS.values())), list(QUESTIONS.values()), list(CHUNKS.values()))["summary"]
    assert s["grounded"] == 1.0                       # it only ever quotes the cited chunk
    assert s["refused_unanswerable"] < 0.5            # but it rarely says "I don't know"
    assert s["correct_same_language"] > s["correct_cross_lingual"]


def test_cli_compares_thresholds_and_writes_json(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("RAG_LLM_BASE_URL", "http://127.0.0.1:9/v1")
    out = tmp_path / "grounding.json"
    assert main(["--extractive", "--retrievers", "bm25", "--min-overlap", "0", "0.5", "--json", str(out)]) == 0
    table = capsys.readouterr().out
    assert table.startswith("| config | correct |") and "| bm25 + extractive, min_overlap=0.5 |" in table
    data = json.loads(out.read_text(encoding="utf-8"))
    assert list(data) == ["bm25 + extractive, min_overlap=0", "bm25 + extractive, min_overlap=0.5"]
    assert len(data["bm25 + extractive, min_overlap=0"]["per_question"]) == 48
    assert data["bm25 + extractive, min_overlap=0.5"]["summary"]["refused_unanswerable"] >= \
        data["bm25 + extractive, min_overlap=0"]["summary"]["refused_unanswerable"]

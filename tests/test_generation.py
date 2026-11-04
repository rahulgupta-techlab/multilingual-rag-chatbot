import json
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from src.evaluation import DOCS
from src.generation import (NO_ANSWER, ExtractiveAnswerer, OpenAICompatibleClient, RAGPipeline, build_messages,
                            extract_citations, main)
from src.ingest import load_corpus
from src.retrieval import BM25Index

CHUNKS = {c.id: c for c in load_corpus(DOCS)}


class FixedRetriever:
    def __init__(self, ids):
        self.ids = ids

    def search(self, query, top_k=5):
        return [(cid, 1.0) for cid in self.ids[:top_k]]


class ScriptedLLM:
    """Returns a fixed reply (or raises it) and records the messages it was sent."""

    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def complete(self, messages):
        self.calls.append(messages)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def _bm25_pipeline(llm=None):
    bm25 = BM25Index()
    bm25.add(list(CHUNKS), [f"{c.title} {c.section} {c.text}" for c in CHUNKS.values()])
    return RAGPipeline(bm25, list(CHUNKS.values()), llm)


def test_prompt_holds_context_ids_and_asks_for_the_users_language():
    chunks = [CHUNKS["bike_rental_en#prices"]]
    system, user = build_messages("Was kostet eine Fahrt?", chunks, "de")
    assert "Always answer in German" in system["content"]
    assert NO_ANSWER["de"] in system["content"]
    assert "[bike_rental_en#prices]" in user["content"] and "2 euros" in user["content"]
    assert user["content"].endswith("Question: Was kostet eine Fahrt?")


def test_citations_keep_order_and_drop_ids_outside_the_context():
    text = "Zwei Euro [bike_rental_en#prices]. Siehe [made_up#x] und [bike_rental_en#prices] [coworking_en#access]."
    assert extract_citations(text, ["coworking_en#access", "bike_rental_en#prices"]) == [
        "bike_rental_en#prices", "coworking_en#access"]


def test_llm_answer_is_used_and_its_citations_are_checked():
    llm = ScriptedLLM("Ohne Abo kostet es 2 Euro pro 30 Minuten [bike_rental_en#prices] [invented#id].")
    pipeline = RAGPipeline(FixedRetriever(["bike_rental_en#prices", "bike_rental_en#opening-hours"]),
                           list(CHUNKS.values()), llm)
    answer = pipeline.ask("Wie viel kostet eine Fahrt ohne Abo?")
    assert answer.mode == "llm" and answer.lang == "de" and answer.fallback_reason is None
    assert answer.citations == ["bike_rental_en#prices"]
    assert answer.context == ["bike_rental_en#prices", "bike_rental_en#opening-hours"]
    assert "Always answer in German" in llm.calls[0][0]["content"]


@pytest.mark.parametrize("error", [urllib.error.URLError("refused"), TimeoutError("slow"), KeyError("choices"),
                                   ValueError("empty completion")])
def test_failed_llm_call_falls_back_to_extractive(error):
    answer = _bm25_pipeline(ScriptedLLM(error)).ask("How much do I pay for a ride without a subscription?")
    assert answer.mode == "extractive"
    assert answer.fallback_reason.startswith(f"LLM call failed: {type(error).__name__}")
    assert answer.text == "Non-subscribers pay 2 euros per 30 minutes from the start of the ride. [bike_rental_en#prices]"
    assert answer.citations == ["bike_rental_en#prices"]


def test_extractive_prefers_rare_terms_and_stays_in_one_section():
    answer = _bm25_pipeline().ask("What is the Wi-Fi password?")
    assert answer.mode == "extractive" and answer.fallback_reason == "no LLM configured"
    assert answer.citations == ["coworking_en#internet"]
    assert '"CoWork-Members"' in answer.text and "first day of every month" in answer.text


def test_extractive_answers_german_question_from_german_document():
    answer = _bm25_pipeline().ask("Kann ich die Mitgliedschaft pausieren, wenn ich krank bin?")
    assert answer.lang == "de" and answer.citations == ["fitnessstudio_de#pausieren"]
    assert "3 Monate" in answer.text


def test_extractive_without_overlap_quotes_the_top_chunk():
    # A cross-lingual question shares no terms with the document; retrieval already picked the chunk.
    text, citations = ExtractiveAnswerer().answer(
        "Wie lange darf ich einen Besprechungsraum reservieren?", [CHUNKS["coworking_en#meeting-rooms"]], "de")
    assert citations == ["coworking_en#meeting-rooms"]
    assert text.startswith("Meeting rooms can be booked in the member portal for up to 4 hours per day.")


def test_no_retrieved_chunks_gives_a_no_answer_in_the_users_language():
    llm = ScriptedLLM("should not be called")
    answer = RAGPipeline(FixedRetriever([]), list(CHUNKS.values()), llm).ask("Wie spät ist es auf dem Mars?")
    assert answer.mode == "no-context" and answer.text == NO_ANSWER["de"] and answer.citations == []
    assert llm.calls == []


class _ChatHandler(BaseHTTPRequestHandler):
    requests: list = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
        reply = json.dumps({"choices": [{"message": {"role": "assistant",
                                                     "content": " 2 Euro [bike_rental_en#prices] "}}]})
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(reply.encode())

    def log_message(self, *args):
        pass


@pytest.fixture
def chat_server():
    _ChatHandler.requests = []
    server = HTTPServer(("127.0.0.1", 0), _ChatHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/v1/"
    server.shutdown()


def test_openai_compatible_client_sends_chat_completions_request(chat_server):
    client = OpenAICompatibleClient(chat_server, model="qwen2.5", api_key="sk-test")
    assert client.complete([{"role": "user", "content": "Preis?"}]) == "2 Euro [bike_rental_en#prices]"
    (req,) = _ChatHandler.requests
    assert req["path"] == "/v1/chat/completions" and req["auth"] == "Bearer sk-test"
    assert req["body"]["model"] == "qwen2.5" and req["body"]["temperature"] == 0.0
    assert req["body"]["messages"] == [{"role": "user", "content": "Preis?"}]


def test_client_is_configured_from_env(monkeypatch):
    monkeypatch.delenv("RAG_LLM_BASE_URL", raising=False)
    assert OpenAICompatibleClient.from_env() is None
    monkeypatch.setenv("RAG_LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("RAG_LLM_MODEL", "llama3.1")
    monkeypatch.delenv("RAG_LLM_API_KEY", raising=False)
    client = OpenAICompatibleClient.from_env()
    assert client.url == "http://localhost:11434/v1/chat/completions"
    assert client.model == "llama3.1" and client.api_key is None


def test_cli_uses_llm_from_env(chat_server, monkeypatch, capsys):
    monkeypatch.setenv("RAG_LLM_BASE_URL", chat_server)
    assert main(["How much does a ride cost?", "--retriever", "bm25"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("2 Euro [bike_rental_en#prices]") and "mode: llm" in out
    assert "sources: bike_rental_en#prices" in out


def test_cli_extractive_flag_ignores_llm(monkeypatch, capsys):
    monkeypatch.setenv("RAG_LLM_BASE_URL", "http://127.0.0.1:9/v1")
    assert main(["What is the Wi-Fi password?", "--retriever", "bm25", "--extractive"]) == 0
    out = capsys.readouterr().out
    assert "mode: extractive (no LLM configured)" in out and "sources: coworking_en#internet" in out

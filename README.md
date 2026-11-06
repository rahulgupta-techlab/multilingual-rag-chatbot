# Multilingual RAG Chatbot (EN / DE)

A retrieval-augmented chatbot that answers questions about documents in **English and German**, in the user's language, **with citations**, including **cross-lingual** questions (ask in German, answer from an English document and vice versa).

The corpus is a small set of self-written rulebooks (bike rental, library, coworking space, gym) so the project runs offline and the answers can be checked by hand.

## Status

Work in progress. It is built in small steps with tests:

- [x] Section-aware markdown chunking with stable chunk IDs for citations
- [x] Lightweight EN/DE language detection for routing answers
- [x] Multilingual embeddings (`paraphrase-multilingual-MiniLM-L12-v2`) and vector search. First check: German question → English doc and vice versa, **4/4 top-1** with the multilingual model vs 2/4 with a lexical baseline
- [x] Hybrid retrieval: BM25 with EN/DE stopwords and umlaut folding, plus dense search, merged with Reciprocal Rank Fusion (rank-based, so BM25 and cosine scores never need calibrating). BM25 catches exact tokens like network names that embeddings miss
- [x] Cross-lingual retrieval benchmark (recall@k, MRR) on 32 labelled EN/DE questions, split into same-language and cross-lingual (see below)
- [x] Answer generation via any OpenAI-compatible LLM (vLLM / Ollama / OpenAI), with an extractive fallback (see below)
- [x] FastAPI chat endpoint and Docker Compose (API + Qdrant + vLLM), with dense vectors stored in Qdrant (see below)
- [x] Groundedness and refusal evaluation ("I don't know" when the documents don't contain the answer), on the 32 benchmark questions plus 16 unanswerable ones (see below)

## Retrieval benchmark

`data/eval/questions.jsonl` holds 32 hand-written questions: one English and one German question for each of the 16 corpus sections, phrased in the user's own words rather than copied from the documents. Half are same-language (EN→EN, DE→DE) and half are cross-lingual (DE question → English document, EN question → German document). `src/evaluation.py` reports recall@1/3/5 and MRR (over the top 10) overall, for each half and for each direction.

```bash
python -m src.evaluation --retrievers bm25 hashing hybrid-hashing
python -m src.evaluation --retrievers multilingual hybrid-multilingual --json results.json  # downloads the model
```

Measured with the offline, lexical retrievers (no model download, same as CI):

| retriever | questions | n | recall@1 | recall@3 | recall@5 | MRR |
|---|---|---|---|---|---|---|
| bm25 | same-language | 16 | 0.625 | 0.812 | 0.812 | 0.708 |
| bm25 | cross-lingual | 16 | 0.125 | 0.250 | 0.250 | 0.177 |
| hashing | same-language | 16 | 0.438 | 0.875 | 0.938 | 0.639 |
| hashing | cross-lingual | 16 | 0.062 | 0.125 | 0.188 | 0.162 |
| hybrid-hashing | same-language | 16 | 0.625 | 0.875 | 1.000 | 0.771 |
| hybrid-hashing | cross-lingual | 16 | 0.125 | 0.250 | 0.250 | 0.226 |

Lexical matching works within one language and mostly fails across languages, which is the gap the multilingual embeddings are there to close.

Multilingual model (`paraphrase-multilingual-MiniLM-L12-v2`), run the second command above to fill in:

| retriever | questions | n | recall@1 | recall@3 | recall@5 | MRR |
|---|---|---|---|---|---|---|
| multilingual | same-language | 16 | | | | |
| multilingual | cross-lingual | 16 | | | | |
| hybrid-multilingual | same-language | 16 | | | | |
| hybrid-multilingual | cross-lingual | 16 | | | | |

## Answer generation

`src/generation.py` retrieves the top chunks for a question and answers it **in the question's language**, citing every chunk it uses as `[chunk-id]`.

- **LLM answerer**: any OpenAI-compatible `/chat/completions` endpoint, configured with `RAG_LLM_BASE_URL` (including the `/v1` prefix), `RAG_LLM_MODEL` and optionally `RAG_LLM_API_KEY`. The prompt contains the retrieved chunks with their ids, tells the model to answer in English or German even when the source is in the other language, to cite only ids from the context, and to reply with a fixed "I don't know" sentence when the context has no answer. Citations to ids that were not in the context are dropped. Temperature is 0.
- **Extractive fallback**: used when no LLM is configured or the call fails (connection error, timeout, malformed or empty response); the answer records why. It quotes the best-matching sentence(s) of one retrieved section, scoring sentences by the IDF of the question terms they contain, and cites that section. It does not translate, so a cross-lingual answer is quoted in the document's language.

```bash
# offline, extractive (no model needed)
python -m src.generation "What is the Wi-Fi password?" --retriever bm25
# The Wi-Fi network is "CoWork-Members". The password changes on the first day of every month and is shown on the screens in the kitchen. [coworking_en#internet]
#
# mode: extractive (no LLM configured)
# sources: coworking_en#internet

# with a local model (Ollama shown; vLLM: http://localhost:8000/v1)
RAG_LLM_BASE_URL=http://localhost:11434/v1 RAG_LLM_MODEL=qwen2.5:7b-instruct \
    python -m src.generation "Wie lange darf ich einen Besprechungsraum buchen?" --retriever hybrid-multilingual
```

The tests cover the prompt, citation checking, every fallback path and the HTTP client against a local stub server, so CI needs no model. Answer quality is measured by the groundedness and refusal evaluation below.

## Groundedness and refusal evaluation

`src/grounding.py` runs the whole pipeline (retrieval + answer) on two question sets:

- the 32 benchmark questions, each now labelled with the **facts** a correct answer must state. A fact is a list of accepted spellings in both languages (`"2 euros"`, `"2 Euro"`, `"zwei Euro"`), so a quoted answer and a translated LLM answer are matched the same way. A test checks that every fact occurs in the section the question is labelled with.
- 16 **unanswerable** questions in `data/eval/unanswerable.jsonl` (8 EN, 8 DE) that sound like they belong to one of the documents but are not answered by any of them: a sauna in the gym, e-books in the library, the price of the bike subscription, Sunday opening hours of the gym (the library's Sunday rule is in the corpus, the gym's is not). The right reply is the fixed "I don't know" sentence.

Every answer is checked for:

| check | definition |
|---|---|
| correct | not a refusal, and every fact appears (case-insensitive, word-bounded, so `2 euro` does not match `12 euro`) |
| cites relevant | at least one citation is a chunk labelled relevant for the question |
| false refusal | an answerable question got "I don't know" |
| refused unanswerable | an unanswerable question got "I don't know" |
| grounded | not a refusal, cites at least one retrieved chunk, cites no id that was not retrieved, and every number in the answer (digits, or number words two to twelve in EN/DE) occurs in a cited chunk. Reported as a share of all non-refusal answers from both sets |

The groundedness check needs no judge model and works across languages, because numbers are the same in both. It targets the most costly error in policy answers (a wrong price, deadline or opening hour) and does not catch a wrong claim without a number.

The extractive fallback now has an optional abstention threshold, `ExtractiveAnswerer(min_overlap=...)`: if the sentences it would quote contain less than that share of the question's terms, it replies "I don't know". The default 0 keeps the old behaviour (always quote).

```bash
python -m src.grounding --extractive --retrievers bm25 hybrid-hashing --min-overlap 0 0.25 0.5 --json grounding.json
```

Measured with the offline retrievers and the **extractive answerer (no LLM)**, same as CI:

| config | correct | same-lang | cross-lingual | cites relevant | false refusals | refused unanswerable | grounded answers |
|---|---:|---:|---:|---:|---:|---:|---:|
| bm25 + extractive, min_overlap=0 | 0.344 | 0.562 | 0.125 | 0.406 | 0.250 | 0.125 | 1.000 (38) |
| bm25 + extractive, min_overlap=0.25 | 0.188 | 0.375 | 0.000 | 0.188 | 0.750 | 0.812 | 1.000 (11) |
| bm25 + extractive, min_overlap=0.5 | 0.031 | 0.062 | 0.000 | 0.031 | 0.969 | 1.000 | 1.000 (1) |
| hybrid-hashing + extractive, min_overlap=0 | 0.344 | 0.562 | 0.125 | 0.406 | 0.000 | 0.000 | 1.000 (48) |
| hybrid-hashing + extractive, min_overlap=0.25 | 0.188 | 0.375 | 0.000 | 0.188 | 0.750 | 0.812 | 1.000 (11) |
| hybrid-hashing + extractive, min_overlap=0.5 | 0.031 | 0.062 | 0.000 | 0.031 | 0.969 | 1.000 | 1.000 (1) |

What this shows:

- Extractive answers are grounded by construction (they quote the cited chunk), so 1.000 here only confirms the check does not flag quotes. The tests feed it scripted LLM answers with an invented number, an invented citation and no citation, and it flags each.
- Without a threshold the extractive answerer almost never says "I don't know": with the hybrid retriever it answers all 16 unanswerable questions with a quote from a related section. BM25's 0.125 refusals (and its 0.250 false refusals) come from questions with no term in common with any document, where retrieval returns nothing.
- Term overlap is a poor abstention signal. At 0.25 it refuses 13 of 16 unanswerable questions but also 75% of the answerable ones, including 15 of the 16 cross-lingual ones, since a German question shares almost no terms with an English document. Telling "not in the documents" apart from "phrased differently" needs a model, which is what the LLM answerer's "I don't know" instruction is for.

With an LLM (any OpenAI-compatible endpoint; the run uses the LLM for every question and falls back to extractive only on errors, which shows up as `extractive` in `summary.modes`):

```bash
RAG_LLM_BASE_URL=http://localhost:8000/v1 RAG_LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct \
    python -m src.grounding --retrievers hybrid-hashing hybrid-multilingual --json grounding-llm.json
```

| config | correct | same-lang | cross-lingual | cites relevant | false refusals | refused unanswerable | grounded answers |
|---|---:|---:|---:|---:|---:|---:|---:|
| hybrid-hashing + llm | | | | | | | |
| hybrid-multilingual + llm | | | | | | | |

## Chat API

`src/api.py` serves the pipeline over HTTP with FastAPI. The index is built once at startup from `data/docs`.

| Endpoint | What it does |
|---|---|
| `POST /chat` | Body: `question` (1–2000 characters), optional `top_k` (1–10, default `RAG_TOP_K` = 3). Returns the `answer`, its `lang`, the `mode` (`llm`, `extractive`, `no-context`), the cited chunk ids (`citations`), the cited chunks with title, section, language and text (`sources`), every chunk the answerer was given (`context`), the `fallback_reason` and `latency_ms`. Blank or too long questions get `422`. |
| `GET /health` | Number of chunks, the retriever, where the vectors live (`qdrant`, `memory`, or `none` for BM25) and the LLM model, if any. |

Configuration is by environment variable: `RAG_RETRIEVER` (any retriever from the benchmark, default `hybrid-hashing`), `RAG_QDRANT_URL` (if set, the dense vectors are stored in and searched from that Qdrant server instead of in memory), `RAG_TOP_K`, `RAG_DOCS` and the `RAG_LLM_*` variables above.

```bash
uvicorn src.api:app --port 8080
curl -X POST localhost:8080/chat -H 'content-type: application/json' \
    -d '{"question": "Kann ich die Mitgliedschaft pausieren, wenn ich krank bin?"}'
# {"answer": "Bei Krankheit oder Schwangerschaft kann die Mitgliedschaft mit ärztlichem Attest für bis zu
#   3 Monate pausiert werden. [fitnessstudio_de#pausieren]", "lang": "de", "mode": "extractive",
#   "citations": ["fitnessstudio_de#pausieren"], "sources": [...], ...}
```

`src/qdrant_store.py` (`QdrantIndex`) has the same interface as the in-memory `VectorIndex`, so it plugs into `HybridRetriever` unchanged. It uses cosine distance, maps each chunk id to a stable UUID5 point id and keeps the chunk id in the payload. A test runs all 32 benchmark questions through both indexes (Qdrant in its local in-memory mode) and checks that the top-5 rankings match. The API tests use FastAPI's `TestClient` and cover the extractive and LLM paths, invented citations, an unreachable LLM, request validation and the Qdrant-backed service.

## Docker Compose

[docker-compose.yml](docker-compose.yml) starts the API on port 8080 and Qdrant on 6333. The API waits for Qdrant to be healthy, writes the chunk vectors into a collection at startup and serves from it. vLLM is optional (profile `vllm`, needs an NVIDIA GPU) and serves `Qwen/Qwen2.5-1.5B-Instruct` on port 8000.

```bash
docker compose up -d --build --wait                  # API + Qdrant, extractive answers
python scripts/smoke.py http://localhost:8080 --vector-store qdrant

# with a local LLM on the GPU
RAG_LLM_BASE_URL=http://vllm:8000/v1 docker compose --profile vllm up -d --build

# multilingual embeddings: adds CPU-only torch and sentence-transformers to the image
MULTILINGUAL=1 RAG_RETRIEVER=hybrid-multilingual docker compose up -d --build
```

By default the image contains only the offline retrievers, so it stays small and needs no model download. CI builds it, starts API and Qdrant with Compose and runs [scripts/smoke.py](scripts/smoke.py), which checks that `/health` reports Qdrant and that one English and one German question come back in their own language with a citation. The `MULTILINGUAL=1` image and the vLLM service are not exercised in CI.

## Run the tests

```bash
pip install -r requirements.txt
python -m pytest
```

## License

Copyright (c) 2026 Rahul Gupta. All rights reserved. Published for viewing and evaluation only; copying or reuse requires written permission. See [LICENSE](LICENSE).

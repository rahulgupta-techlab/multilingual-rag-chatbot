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
- [ ] FastAPI chat endpoint and Docker Compose (API + Qdrant + vLLM)
- [ ] Groundedness and refusal evaluation ("I don't know" when the documents don't contain the answer)

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

The tests cover the prompt, citation checking, every fallback path and the HTTP client against a local stub server, so CI needs no model. Answer quality with a real LLM is not measured yet; that is the groundedness and refusal evaluation on the checklist.

## Run the tests

```bash
pip install -r requirements.txt
python -m pytest
```

## License

Copyright (c) 2026 Rahul Gupta. All rights reserved. Published for viewing and evaluation only; copying or reuse requires written permission. See [LICENSE](LICENSE).

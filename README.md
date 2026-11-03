# Multilingual RAG Chatbot (EN / DE)

A retrieval-augmented chatbot that answers questions about policy documents in **English and German**, in the user's language, **with citations**, including **cross-lingual** questions (ask in German, answer from an English document and vice versa).

The corpus is a small set of self-written rulebooks (bike rental, library, coworking space, gym) so the project runs offline and the answers can be checked by hand.

## Status

Work in progress. It is built in small steps with tests:

- [x] Section-aware markdown chunking with stable chunk IDs for citations
- [x] Lightweight EN/DE language detection for routing answers
- [x] Multilingual embeddings (`paraphrase-multilingual-MiniLM-L12-v2`) and vector search. First check: German question → English doc and vice versa, **4/4 top-1** with the multilingual model vs 2/4 with a lexical baseline
- [x] Hybrid retrieval: BM25 with EN/DE stopwords and umlaut folding, plus dense search, merged with Reciprocal Rank Fusion (rank-based, so BM25 and cosine scores never need calibrating). BM25 catches exact tokens like network names that embeddings miss
- [x] Cross-lingual retrieval benchmark (recall@k, MRR) on 32 labelled EN/DE questions, split into same-language and cross-lingual (see below)
- [ ] Answer generation via any OpenAI-compatible LLM (vLLM / Ollama / OpenAI), with an extractive fallback
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

## Run the tests

```bash
pip install -r requirements.txt
python -m pytest
```

## License

Copyright (c) 2026 Rahul Gupta. All rights reserved. Published for viewing and evaluation only; copying or reuse requires written permission. See [LICENSE](LICENSE).
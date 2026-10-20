# Multilingual RAG Chatbot (EN / DE)

A retrieval-augmented chatbot that answers questions about policy documents in **English and German**, in the user's language, **with citations**, including **cross-lingual** questions (ask in German, answer from an English document and vice versa).

The corpus is a small set of self-written rulebooks (bike rental, library, coworking space, gym) so the project runs offline and the answers can be checked by hand.

## Status

Work in progress. It is built in small steps with tests:

- [x] Section-aware markdown chunking with stable chunk IDs for citations
- [x] Lightweight EN/DE language detection for routing answers
- [x] Multilingual embeddings (`paraphrase-multilingual-MiniLM-L12-v2`) and vector search. First check: German question → English doc and vice versa, **4/4 top-1** with the multilingual model vs 2/4 with a lexical baseline
- [x] Hybrid retrieval: BM25 with EN/DE stopwords and umlaut folding, plus dense search, merged with Reciprocal Rank Fusion (rank-based, so BM25 and cosine scores never need calibrating). BM25 catches exact tokens like network names that embeddings miss
- [ ] Cross-lingual retrieval benchmark (recall@k, MRR) on labelled EN/DE questions
- [ ] Answer generation via any OpenAI-compatible LLM (vLLM / Ollama / OpenAI), with an extractive fallback
- [ ] FastAPI chat endpoint and Docker Compose (API + Qdrant + vLLM)
- [ ] Groundedness and refusal evaluation ("I don't know" when the documents don't contain the answer)

## Run the tests

```bash
pip install -r requirements.txt
python -m pytest
```

## License

MIT

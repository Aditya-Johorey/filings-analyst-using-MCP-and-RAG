# Filings Analyst: Agentic RAG over SEC 10-K Filings

A research assistant that answers questions about public company filings with **cited, verifiable answers**. It uses an agentic RAG loop that checks its own retrieval, rewrites weak queries, and only then answers. The whole pipeline is also exposed as an **MCP server**, so other agents (Claude Agent SDK, CrewAI, etc.) can use it as a research tool.

> **Disclaimer:** This is a research aid, not financial advice. Always verify figures against the original filing.

---

## Why this exists

- Plain LLMs hallucinate numbers.
- Basic RAG fails silently when retrieval returns the wrong chunks.
- Other AI agents have no clean, reliable way to reuse a research pipeline as a tool.

## How it works

```
retrieve --> grade --relevant--------------> generate --> answer + citations
               |
               +--not relevant--> reformulate --> retrieve   (max 2 retries)
```

1. **Retrieve:** vector search in Qdrant, optionally filtered by `company` and `year`.
2. **Grade:** the LLM checks whether the chunks contain the facts needed to answer.
3. **Reformulate:** if not, the query is rewritten using filing terminology and the grader's feedback.
4. **Generate:** the answer uses only the retrieved context, copies figures verbatim with units and periods, and cites sources like `[1]`.

The response includes a `grounded` flag. It is `False` when retries ran out and the answer is low confidence.

## Tech stack

| Layer | Technology |
|---|---|
| Orchestration | LangChain, LangGraph |
| LLM | Groq free tier (`llama-3.1-8b-instant`), Ollama optional for local use |
| Embeddings | FastEmbed (`BAAI/bge-small-en-v1.5`, runs on CPU) |
| Vector DB | Qdrant Cloud (free tier) |
| Backend | FastAPI |
| Agent interface | MCP Python SDK (FastMCP) |
| UI | Tailwind CSS |
| Tracing and evals | LangSmith |
| Hosting | Hugging Face Spaces (free) |
| Data | SEC EDGAR 10-K filings |

## Project structure

```
filings-analyst/
├── app/
│   ├── __init__.py
│   ├── config.py        # LLM, embeddings, Qdrant client factories
│   ├── download.py      # fetch 10-Ks from SEC EDGAR
│   ├── ingest.py        # parse, chunk, embed, upload to Qdrant
│   ├── graph.py         # LangGraph agentic loop
│   ├── api.py           # FastAPI endpoints          (in progress)
│   └── mcp_server.py    # MCP tool interface         (planned)
├── static/
│   └── index.html       # Tailwind chat UI           (in progress)
├── data/                # downloaded filings (git-ignored)
├── test_filter.py       # verifies metadata filtering
├── test_graph.py        # smoke test for the agent
├── requirements.txt
├── .env                 # secrets (git-ignored)
└── README.md
```

## Setup

### 1. Get free API keys

- **Qdrant Cloud:** create a free cluster at [cloud.qdrant.io](https://cloud.qdrant.io), then copy the cluster URL and API key.
- **Groq:** [console.groq.com](https://console.groq.com)
- **LangSmith:** [smith.langchain.com](https://smith.langchain.com)

### 2. Install

```bash
git clone <your-repo-url>
cd filings-analyst
python -m venv venv
# Windows PowerShell:  venv\Scripts\activate
# macOS/Linux:         source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure `.env`

```
QDRANT_URL=https://xxxx.cloud.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_key
GROQ_API_KEY=your_groq_key
LLM_PROVIDER=groq

LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=filings-analyst

SEC_NAME=Your Name
SEC_EMAIL=you@email.com
```

`SEC_NAME` and `SEC_EMAIL` are required by SEC EDGAR to identify automated clients. They are only used by `download.py`, so you do not need them in your hosting environment.

### 4. Download and index the filings (one time, from your laptop)

```bash
python -m app.download     # fetches 10-Ks for AAPL, MSFT, TSLA
python -m app.ingest       # chunks, embeds, uploads to Qdrant Cloud
```

The deployed app only queries Qdrant. It never ingests.

### 5. Verify

```bash
python test_filter.py      # should print only AAPL 10-K sources
python test_graph.py       # runs the agent end to end
```

## Usage

### Python

```python
from app.graph import ask

result = ask("What were Apple's total net sales?", company="AAPL", year="2025")
print(result["answer"])
print(result["sources"])
print(result["retries"], result["grounded"])
```

### Response shape

```json
{
  "answer": "Total net sales were $X million [1] ...",
  "sources": [{"id": 1, "source": "AAPL 10-K 2025", "snippet": "..."}],
  "retries": 0,
  "grounded": true
}
```

### REST API and MCP tool

Documented here once those parts are built.

## Notes and known limitations

- `year` in metadata is the **filing year** taken from the accession number, not the fiscal year. Tesla's FY2023 10-K, for example, was filed in 2024.
- The 8B model used on the free tier occasionally returns malformed structured output. Rerunning usually fixes it.
- Groq's free tier is rate limited. `MAX_RETRIES` is capped at 2 to protect it.
- Embedding model changes require re-running `python -m app.ingest` into a fresh collection, since vectors from different models are not compatible.

## Roadmap

- [x] Project setup and config
- [x] SEC filings download
- [x] Ingestion with `company`, `form`, `year` metadata
- [x] Qdrant Cloud with indexed metadata filters
- [x] LangGraph agentic loop with grading, retry, and citations
- [ ] FastAPI endpoints
- [ ] Tailwind chat UI
- [ ] MCP server with a documented tool schema and error responses
- [ ] LangSmith evaluation dataset and retrieval-quality metrics
- [ ] Deployment to Hugging Face Spaces

## Troubleshooting

| Problem | Fix |
|---|---|
| `404 Collection filings doesn't exist` | Run `python -m app.ingest` and confirm it prints a chunk count |
| Download returns nothing or 403 | Check `SEC_NAME` and `SEC_EMAIL` in `.env` |
| Groq `429` errors | Wait a minute and retry, since you hit the free-tier limit |
| Empty search results | Confirm ingest finished and the collection has points |

## License

MIT, or your choice. Filing data comes from SEC EDGAR and is public.
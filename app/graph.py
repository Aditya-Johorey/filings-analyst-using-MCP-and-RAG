# app/graph.py
import re
from typing import TypedDict, List, Optional
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from qdrant_client import models
from app.config import get_llm, get_store

llm, store = get_llm(), get_store()
MAX_RETRIES = 2

class State(TypedDict):
    question: str
    query: str
    company: Optional[str]
    year: Optional[str]
    docs: List[Document]
    relevant: bool
    feedback: str
    attempts: int
    answer: str

class Grade(BaseModel):
    relevant: bool = Field(description="True only if the context contains the specific facts or figures needed")
    reason: str = Field(description="One sentence: what is missing if not relevant")

def build_filter(company, year):
    must = []
    if company:
        must.append(models.FieldCondition(
            key="metadata.company", match=models.MatchValue(value=company.upper())))
    if year:
        must.append(models.FieldCondition(
            key="metadata.year", match=models.MatchValue(value=str(year))))
    return models.Filter(must=must) if must else None

def retrieve(s: State):
    docs = store.similarity_search(
        s["query"], k=6, filter=build_filter(s["company"], s["year"]))
    return {"docs": docs}

def grade(s: State):
    if not s["docs"]:
        return {"relevant": False, "feedback": "No documents were returned."}
    ctx = "\n\n".join(d.page_content for d in s["docs"])
    g = llm.with_structured_output(Grade).invoke(
        "You are grading search results from SEC filings.\n"
        f"Question: {s['question']}\n\nContext:\n{ctx}\n\n"
        "Does the context contain the specific information needed to answer?")
    return {"relevant": g.relevant, "feedback": g.reason}

def reformulate(s: State):
    q = llm.invoke(
        "Rewrite this search query to retrieve better passages from a 10-K filing. "
        "Never add company names, tickers or years that are not in the original question.\n"
        "Use terminology filings actually use (e.g. 'net sales', 'total revenues', "
        "'risk factors', 'operating income').\n"
        f"Question: {s['question']}\nFailed query: {s['query']}\n"
        f"Why it failed: {s['feedback']}\n"
        "Return only the new query.").content.strip()
    return {"query": q, "attempts": s["attempts"] + 1}

def generate(s: State):
    ctx = "\n\n".join(
        f"[{i+1}] {d.metadata['source']}\n{d.page_content}"
        for i, d in enumerate(s["docs"]))
    prompt = ("You are a financial filings research assistant.\n"
              "Rules:\n"
              "1. Answer ONLY from the numbered context below.\n"
              "2. Copy figures exactly as written, with their unit (millions, billions) and period.\n"
              "3. Cite every claim with plain square brackets like [1] or [2]. Never use any other bracket style.\n"
              "4. If the context does not contain the answer, say so. Never guess.\n"
              "5. If the question is ambiguous (no company or period named), say what is unclear instead of choosing for the user.\n\n"
              f"Context:\n{ctx}\n\nQuestion: {s['question']}")
    ans = llm.invoke(prompt).content.strip()
    if not ans:
        ans = llm.invoke(prompt).content.strip()
    ans = re.sub(r"[【\[]\s*(\d+)\s*[】\]]", r"[\1]", ans)   # normalize 【3】 -> [3]
    return {"answer": ans or "The model returned an empty response. Please retry."}

def route(s: State):
    if s["relevant"] or s["attempts"] >= MAX_RETRIES:
        return "generate"
    return "reformulate"

# ---- graph wiring: all edges must come BEFORE compile() ----
g = StateGraph(State)
g.add_node("retrieve", retrieve)
g.add_node("grade", grade)
g.add_node("reformulate", reformulate)
g.add_node("generate", generate)

g.set_entry_point("retrieve")
g.add_edge("retrieve", "grade")
g.add_conditional_edges("grade", route, ["generate", "reformulate"])
g.add_edge("reformulate", "retrieve")
g.add_edge("generate", END)

graph = g.compile()

def ask(question: str, company: str | None = None, year: str | None = None) -> dict:
    out = graph.invoke({
        "question": question, "query": question,
        "company": company, "year": year,
        "docs": [], "relevant": False, "feedback": "",
        "attempts": 0, "answer": ""})
    return {
        "answer": out["answer"],
        "sources": [{"id": i + 1,
                     "source": d.metadata["source"],
                     "snippet": d.page_content[:300]}
                    for i, d in enumerate(out["docs"])],
        "retries": out["attempts"],
        "grounded": out["relevant"],
    }
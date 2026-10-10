# debug_graph.py
from app.graph import graph

q = "What were Apple's total net sales?"
init = {"question": q, "query": q, "company": "AAPL", "year": "2025",
        "docs": [], "relevant": False, "feedback": "",
        "attempts": 0, "answer": ""}

for step in graph.stream(init, stream_mode="updates"):
    for node, upd in step.items():
        print("==", node)
        for k, v in upd.items():
            print(f"   {k}:", len(v) if k == "docs" else repr(str(v)[:300]))
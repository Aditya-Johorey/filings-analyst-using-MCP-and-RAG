# test_graph.py
from app.graph import ask
import json

r = ask("What were Apple's total net sales?", company="AAPL", year="2025")
print(json.dumps(r, indent=2)[:2500])

r = ask("What are Tesla's main supply chain risks?", company="TSLA")
print(r["answer"], "\nretries:", r["retries"], "grounded:", r["grounded"])
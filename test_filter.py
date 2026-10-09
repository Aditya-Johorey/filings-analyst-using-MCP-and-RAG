from qdrant_client import models
from app.config import get_store

f = models.Filter(must = [models.FieldCondition(
    key = "metadata.company", match = models.MatchValue(value = "AAPL")
)])
for d in get_store().similarity_search("risk factors", k=3, filter = f):
    print(d.metadata["source"])
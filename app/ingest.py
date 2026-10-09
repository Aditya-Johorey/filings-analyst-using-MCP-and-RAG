import os, re
from pathlib import Path
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from app.config import COLLECTION, get_embeddings, get_client

def load_filings():
    docs = []
    for p in Path("data/sec-edgar-filings").glob("*/*/*/primary-document.html"):
        ticker, form, accession = p.parts[-4], p.parts[-3], p.parts[-2]
        soup = BeautifulSoup(p.read_text(encoding="utf-8", errors="ignore"), "lxml")
        for t in soup(["script", "style", "ix:header"]):  # drop hidden XBRL junk
            t.decompose()
        text = re.sub(r"\n\s*\n+", "\n\n", soup.get_text("\n"))
        year = "20" + accession.split("-")[1]   # filing year
        docs.append(Document(
            page_content=text,
            metadata={"source": f"{ticker} {form} {year}",
                      "company": ticker, "form": form, "year": year}))
    return docs

if __name__ == "__main__":
    docs = load_filings()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 150).split_documents(docs)

    QdrantVectorStore.from_documents(
        chunks, get_embeddings(),
        collection_name = COLLECTION,
        url = os.environ["QDRANT_URL"],
        api_key = os.getenv("QDRANT_API_KEY"),
        batch_size = 24
    )

    #index metadata fields so filtered searches stay fast
    client = get_client()
    for field in ["company", "form", "year"]:
        client.create_payload_index(COLLECTION, f"metadata.{field}", "keyword")

    print(f"Indexed {len(chunks)} chunks from {len(docs)} filings")
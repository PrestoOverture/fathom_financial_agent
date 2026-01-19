import os
import hashlib
import json
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import make_url

from llama_parse import LlamaParse
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding

ROOT_DIR = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT_DIR / "data" / "pdfs"
CACHE_DIR = ROOT_DIR / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(override=True)

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

def get_file_hash(file_path: Path) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()

def parse_pdf(file_path: Path) -> list[Document]:
    file_hash = get_file_hash(file_path)
    cache_path = CACHE_DIR / f"{file_hash}.json"

    if cache_path.exists():
        print(f"Loading cached documents from {cache_path}")
        with open(cache_path, "r") as f:
            data = json.load(f)
        return [Document(text=item["text"], metadata=item["metadata"]) for item in data]
    else:
        print(f"Parsing PDF {file_path}")
        parser = LlamaParse(result_type="markdown", verbose=True, language="en")
        documents = parser.load_data(str(file_path))
        cache_data = [{"text": doc.text, "metadata": doc.metadata} for doc in documents]
        with open(cache_path, "w") as f:
            json.dump(cache_data, f, indent=2, default=str)
        return documents 

def ingest_to_postgres(documents: list[Document]):
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError("DATABASE_URL is not set")
    else:
        print("DATABASE_URL is set")

    database_url = make_url(database_url)
    port = database_url.port or 5432

    vector_store = PGVectorStore.from_params(
        host=database_url.host,
        port=port,
        user=database_url.username,
        password=database_url.password,
        database=database_url.database,
        table_name="financial_docs",
        embed_dim=1536,
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    node_parser = MarkdownElementNodeParser()
    print("Waiting for processing nodes. Please be patient...")

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=[node_parser],
        show_progress=True
    )
    print("Ingestion done!")

if __name__ == "__main__":
    test_file = PDF_DIR / "2023-03-09_Ulta_Beauty_Announces_Fourth_Quarter_Fiscal_2022_164.pdf"

    if test_file.exists():
        docs = parse_pdf(test_file)
        ingest_to_postgres(docs)
    else:
        print("File not found.")
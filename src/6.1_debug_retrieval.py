from dotenv import load_dotenv
import os
import re
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from sqlalchemy import make_url
import ast

load_dotenv(override=True)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")


def check_table_integrity():
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

    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    retriever = index.as_retriever(similarity_top_k=5)

    query = "Total Assets"
    print(f"Querying: {query}")
    results = retriever.retrieve(query)

    if not results:
        print("No results found")
    else:
        print(f"Found {len(results)} results")
        for i, node in enumerate(results):
            content = node.node.get_content()
            score = node.score
            content_to_check = content
            if content.strip().startswith("{") and "'text':" in content:
                node_dict = ast.literal_eval(content)
                if "text" in node_dict:
                    content_to_check = node_dict["text"]
            has_table = bool(re.search(r"\|\s*-{3,}", content_to_check))
            print(f"Result {i+1}:")
            print(f"Score: {score:.4f}")
            print(f"Has Table: {'Yes' if has_table else 'No'}")
            print(f"Content: {content_to_check}")
            print("-" * 100)

if __name__ == "__main__":
    check_table_integrity()
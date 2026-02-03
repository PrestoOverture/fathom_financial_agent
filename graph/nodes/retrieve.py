import ast
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from graph.state import AgentState
from sqlalchemy import make_url

load_dotenv(override=True)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")


def parse_node_content(raw_content: str) -> tuple[str, dict]:
    """
    Parse node content which may be a serialized dict with 'text' key.
    Returns (text_content, extracted_metadata).
    """
    content = raw_content
    extracted_metadata = {}

    # Check if content is a serialized dictionary
    if raw_content.strip().startswith("{") and "'text':" in raw_content:
        try:
            node_dict = ast.literal_eval(raw_content)
            if isinstance(node_dict, dict):
                # Extract the actual text
                if "text" in node_dict:
                    content = node_dict["text"]
                # Extract metadata from the dict if present
                if "metadata" in node_dict and isinstance(node_dict["metadata"], dict):
                    extracted_metadata = node_dict["metadata"]
        except (ValueError, SyntaxError):
            # If parsing fails, use raw content
            pass

    return content, extracted_metadata


def retrieve(state: AgentState):
    print(f"Retrieving question: {state.question}")
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
    nodes = retriever.retrieve(state.question)

    retrieved_data = []
    evidences = []
    for node in nodes:
        raw_content = node.node.get_content()
        node_metadata = node.node.metadata or {}
        score = node.score if node.score else 0.0

        # Parse content (may be serialized dict)
        text_content, extracted_metadata = parse_node_content(raw_content)

        # Merge metadata: prefer node.metadata, fall back to extracted
        merged_metadata = {**extracted_metadata, **node_metadata}
        doc_name = merged_metadata.get("doc_name", "Unknown")

        retrieved_data.append({
            "text": text_content,
            "metadata": merged_metadata,
            "score": score,
        })

        evidences.append(f"Source: {doc_name}\nContent: {text_content}")

    full_evidence = "\n\n---\n\n".join(evidences)
    return {
        "retrieved_nodes": retrieved_data,
        "evidence": full_evidence,
    }

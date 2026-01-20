import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from graph.state import AgentState
from sqlalchemy import make_url

load_dotenv(override=True)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

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
    contexts = []
    for node in nodes: 
        node_content = node.node.get_content()
        node_metadata = node.node.metadata
        score = node.score if node.score else 0.0

        retrieved_data.append({
            "text": node_content,
            "metadata": node_metadata,
            "score": score,
        })

        doc_name = node_metadata.get("doc_name", "Unknown")
        contexts.append(f"Source: {doc_name}\nContent: {node_content}")
    
    full_context = "\n\n---\n\n".join(contexts)
    return {
        "retrieved_nodes": retrieved_data,
        "context": full_context,
    }
import ast
import os
import re
from typing import Any

from dotenv import load_dotenv
from graph.state import AgentState
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.vector_stores import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import make_url

load_dotenv(override=True)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# Module-level singleton to avoid reloading on every request.
_reranker = SentenceTransformerRerank(
    model="BAAI/bge-reranker-base",
    top_n=5,
)

COMPANY_ALIASES: dict[str, str] = {
    "coca cola": "COCACOLA",
    "coca-cola": "COCACOLA",
    "coke": "COCACOLA",
    "amazon": "AMAZON",
    "verizon": "VERIZON",
    "pfizer": "PFIZER",
    "lockheed martin": "LOCKHEEDMARTIN",
    "lockheed": "LOCKHEEDMARTIN",
    "mgm resorts": "MGMRESORTS",
    "mgm": "MGMRESORTS",
    "amcor": "AMCOR",
    "ulta beauty": "ULTA",
    "ulta": "ULTA",
    "3m": "3M",
    "jnj": "JNJ",
    "johnson & johnson": "JNJ",
    "johnson and johnson": "JNJ",
    "j&j": "JNJ",
    "amd": "AMD",
    "boeing": "BOEING",
}


def resolve_company(question: str) -> str | None:
    """Extract company from question using alias matching."""
    q_lower = question.lower()
    for alias in sorted(COMPANY_ALIASES.keys(), key=len, reverse=True):
        if alias in q_lower:
            return COMPANY_ALIASES[alias]
    return None


def resolve_fiscal_year(question: str) -> str | None:
    """Extract fiscal year from question text."""
    q_lower = question.lower()

    match = re.search(r"(?:fy|fiscal\s*(?:year)?\s*|q[1-4]\s*)(\d{4})", q_lower)
    if match:
        return match.group(1)

    match = re.search(r"fy\s*(\d{2})\b", q_lower)
    if match:
        return "20" + match.group(1)

    match = re.search(r"\b(20[1-2]\d)\b", q_lower)
    if match:
        return match.group(1)

    return None


def parse_node_content(raw_content: str) -> tuple[str, dict[str, Any]]:
    content = raw_content
    extracted_metadata: dict[str, Any] = {}

    if raw_content.strip().startswith("{") and "'text':" in raw_content:
        try:
            node_dict = ast.literal_eval(raw_content)
            if isinstance(node_dict, dict):
                if "text" in node_dict:
                    content = node_dict["text"]
                if "metadata" in node_dict and isinstance(node_dict["metadata"], dict):
                    extracted_metadata = node_dict["metadata"]
        except (ValueError, SyntaxError):
            pass

    return content, extracted_metadata


def retrieve(state: AgentState) -> dict[str, Any]:
    print(f"Retrieving for: {state.question}")

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL is not set")

    url = make_url(database_url)
    vector_store = PGVectorStore.from_params(
        host=url.host,
        port=url.port or 5432,
        user=url.username,
        password=url.password,
        database=url.database,
        table_name="financial_docs",
        embed_dim=1536,
    )
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    company = resolve_company(state.question)
    fiscal_year = resolve_fiscal_year(state.question)

    filter_list: list[MetadataFilter] = []
    if company:
        print(f"  Resolved company: {company}")
        filter_list.append(
            MetadataFilter(key="company", value=company, operator=FilterOperator.EQ)
        )
    else:
        print("  Resolved company: None")

    if fiscal_year:
        print(f"  Resolved fiscal_year: {fiscal_year}")
        filter_list.append(
            MetadataFilter(
                key="fiscal_year",
                value=fiscal_year,
                operator=FilterOperator.EQ,
            )
        )
    else:
        print("  Resolved fiscal_year: None")

    filters = MetadataFilters(filters=filter_list) if filter_list else None
    retriever = index.as_retriever(similarity_top_k=15, filters=filters)
    nodes = retriever.retrieve(state.question)

    if not nodes and filters and company and fiscal_year:
        print("  WARNING: company+year filter returned 0 results, retrying company-only")
        company_only_filter = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="company",
                    value=company,
                    operator=FilterOperator.EQ,
                )
            ]
        )
        fallback_retriever = index.as_retriever(
            similarity_top_k=15,
            filters=company_only_filter,
        )
        nodes = fallback_retriever.retrieve(state.question)

    if not nodes and filters:
        print("  WARNING: All filtered retrieval returned 0 results, retrying without filters")
        fallback_retriever = index.as_retriever(similarity_top_k=15)
        nodes = fallback_retriever.retrieve(state.question)

    if nodes:
        print(f"  Retrieved {len(nodes)} candidates before reranking")
        nodes = _reranker.postprocess_nodes(nodes, query_str=state.question)
        print(f"  Kept {len(nodes)} nodes after reranking")
    else:
        print("  Retrieved 0 nodes after fallback sequence")

    retrieved_data: list[dict[str, Any]] = []
    evidences: list[str] = []

    for node in nodes:
        raw_content = node.node.get_content()
        node_metadata = node.node.metadata or {}
        score = node.score if node.score else 0.0

        text_content, extracted_metadata = parse_node_content(raw_content)
        merged_metadata = {**extracted_metadata, **node_metadata}
        doc_name = merged_metadata.get("doc_name", "Unknown")

        retrieved_data.append(
            {
                "text": text_content,
                "metadata": merged_metadata,
                "score": score,
            }
        )
        evidences.append(f"Source: {doc_name}\nContent: {text_content}")

    full_evidence = "\n\n---\n\n".join(evidences)
    return {
        "retrieved_nodes": retrieved_data,
        "evidence": full_evidence,
    }

import ast
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from graph.nodes.retrieve import resolve_company, resolve_fiscal_year
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.vector_stores import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import make_url

ROOT_DIR = Path(__file__).resolve().parents[1]
TEST_FILE = ROOT_DIR / "data" / "test" / "test.jsonl"
OUTPUT_FILE = ROOT_DIR / "results" / "test_set" / "retrieval_context.json"
TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
USE_FILTERING = True
USE_RERANKER = True
RERANKER_TOP_N = 5

_reranker = None
if USE_RERANKER:
    from llama_index.postprocessor.sbert_rerank import (
        SentenceTransformerRerank,
    )

    _reranker = SentenceTransformerRerank(
        model="BAAI/bge-reranker-base",
        top_n=RERANKER_TOP_N,
    )

load_dotenv(override=True)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")


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


def load_questions(path: Path) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            questions.append(json.loads(line))
    return questions


def build_retriever(
    similarity_top_k: int,
    filters: MetadataFilters | None = None,
):
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
    return index.as_retriever(similarity_top_k=similarity_top_k, filters=filters)


def retrieve_contexts(
    questions: list[dict[str, Any]],
    similarity_top_k: int,
) -> dict[str, Any]:
    retrievals: list[dict[str, Any]] = []
    for idx, item in enumerate(questions, start=1):
        question = item.get("question", "")
        financebench_id = item.get("financebench_id")
        expected_doc_name = item.get("metadata", {}).get("doc_name")

        print(f"[{idx}/{len(questions)}] Retrieving: {financebench_id}")
        company: str | None = None
        fiscal_year: str | None = None
        filters: MetadataFilters | None = None

        if USE_FILTERING:
            company = resolve_company(question)
            fiscal_year = resolve_fiscal_year(question)

            print(f"  Resolved company: {company}")
            print(f"  Resolved fiscal_year: {fiscal_year}")

            filter_list: list[MetadataFilter] = []
            if company:
                filter_list.append(
                    MetadataFilter(
                        key="company",
                        value=company,
                        operator=FilterOperator.EQ,
                    )
                )
            if fiscal_year:
                filter_list.append(
                    MetadataFilter(
                        key="fiscal_year",
                        value=fiscal_year,
                        operator=FilterOperator.EQ,
                    )
                )
            filters = MetadataFilters(filters=filter_list) if filter_list else None

        retriever = build_retriever(similarity_top_k, filters=filters)
        nodes = retriever.retrieve(question)

        if USE_FILTERING and not nodes and filters and company and fiscal_year:
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
            fallback_retriever = build_retriever(
                similarity_top_k,
                filters=company_only_filter,
            )
            nodes = fallback_retriever.retrieve(question)

        if USE_FILTERING and not nodes and filters:
            print("  WARNING: All filtered retrieval returned 0 results, retrying without filters")
            fallback_retriever = build_retriever(similarity_top_k)
            nodes = fallback_retriever.retrieve(question)

        if USE_RERANKER and _reranker is not None and nodes:
            nodes = _reranker.postprocess_nodes(nodes, query_str=question)

        # Evaluator has no internal limit; always store only final top-5 chunks.
        nodes = nodes[:RERANKER_TOP_N]

        chunks: list[dict[str, Any]] = []
        for node in nodes:
            raw_content = node.node.get_content()
            node_metadata = node.node.metadata or {}
            score = float(node.score) if node.score is not None else 0.0

            text_content, extracted_metadata = parse_node_content(raw_content)
            merged_metadata = {**extracted_metadata, **node_metadata}

            chunks.append(
                {
                    "text": text_content,
                    "metadata": merged_metadata,
                    "score": score,
                }
            )

        retrievals.append(
            {
                "financebench_id": financebench_id,
                "question": question,
                "doc_name": expected_doc_name,
                "retrieved_chunks": chunks,
            }
        )

    return {
        "source_file": str(TEST_FILE),
        "total_questions": len(questions),
        "similarity_top_k": similarity_top_k,
        "use_filtering": USE_FILTERING,
        "use_reranker": USE_RERANKER,
        "final_top_n": RERANKER_TOP_N,
        "retrievals": retrievals,
    }


def main() -> None:
    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Missing test file: {TEST_FILE}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    questions = load_questions(TEST_FILE)
    results = retrieve_contexts(questions, TOP_K)

    with OUTPUT_FILE.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)

    print(f"Wrote retrieval contexts to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

import asyncio
import json
from datetime import datetime, timezone
from typing import AsyncGenerator


def format_sse(event_type: str, data: dict) -> str:
    data["event_type"] = event_type
    data["timestamp"] = datetime.now(timezone.utc).isoformat()
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


async def stream_graph_sse(
    graph,
    question: str,
    request_id: str,
) -> AsyncGenerator[str, None]:
    try:
        yield format_sse("graph_start", {"request_id": request_id, "question": question})

        final_state = {}

        async for mode, data in graph.astream(
            {"question": question},
            stream_mode=["updates", "custom"],
        ):
            if mode == "custom":
                if data.get("event") == "reasoning_delta":
                    yield format_sse(
                        "reasoning_delta",
                        {"request_id": request_id, "delta": data.get("delta", "")},
                    )

            elif mode == "updates":
                for node_name, output in data.items():
                    if node_name.startswith("__"):
                        continue

                    if node_name == "retrieve":
                        nodes = output.get("retrieved_nodes", [])
                        sources = list(
                            set(
                                n.get("metadata", {}).get("doc_name", "unknown")
                                for n in nodes
                            )
                        )
                        yield format_sse(
                            "retrieve_update",
                            {
                                "request_id": request_id,
                                "source_count": len(nodes),
                                "sources": sources,
                            },
                        )

                    elif node_name == "reason":
                        answer = output.get("answer", "")
                        final_state["answer"] = answer
                        yield format_sse(
                            "reason_update",
                            {"request_id": request_id, "answer": answer},
                        )

                    elif node_name == "verify":
                        errors_found = output.get("arithmetic_errors_found", False)
                        log = output.get("verification_log", [])
                        final_state["has_errors"] = errors_found
                        yield format_sse(
                            "verify_update",
                            {
                                "request_id": request_id,
                                "arithmetic_errors_found": errors_found,
                                "error_count": len(log),
                            },
                        )

        yield format_sse(
            "graph_complete",
            {
                "request_id": request_id,
                "answer": final_state.get("answer", ""),
                "has_errors": final_state.get("has_errors", False),
            },
        )

    except asyncio.CancelledError:
        # if client disconnected, then exit
        return
    except Exception as e:
        # if an error occurs, then emit an error event
        yield format_sse(
            "error",
            {
                "request_id": request_id,
                "error_type": type(e).__name__,
                "message": str(e),
            },
        )

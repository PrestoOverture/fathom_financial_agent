import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from api.schemas import QueryRequest
from api.sse import stream_graph_sse
from graph.workflow import create_graph

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.graph = create_graph()
    yield

app = FastAPI(title="Fathom Financial Agent API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.post("/query")
async def query_stream(request: Request, body: QueryRequest) -> StreamingResponse:
    request_id = str(uuid.uuid4())
    graph = request.app.state.graph

    async def wrapped_stream():
        async for event in stream_graph_sse(graph, body.question, request_id):
            if await request.is_disconnected():
                break
            yield event

    return StreamingResponse(
        wrapped_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.get("/health")
async def health():
    return {"status": "healthy"}

# streaming_server.py
import sys

sys.path.insert(0, "/home/mujahid/PycharmProjects/rag-anything-test/raganything-source")
sys.path.insert(
    0,
    "/home/mujahid/PycharmProjects/rag-anything-test/venv/lib/python3.11/site-packages",
)
import os
import asyncio
from typing import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc


class QueryRequest(BaseModel):
    query: str
    mode: str = "hybrid"
    stream: bool = False


rag: RAGAnything = None


def create_llm_func(api_key: str, stream: bool = False):
    """Create LLM function with optional streaming"""

    async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        # Enable streaming if requested
        if stream:
            kwargs["stream"] = True
            # For streaming, need to include stream_options for usage tracking
            kwargs["stream_options"] = {"include_usage": True}

        return await openai_complete_if_cache(
            "gpt-4o-mini",
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
            **kwargs,
        )

    return llm_model_func


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag
    api_key = os.getenv("OPENAI_API_KEY")

    config = RAGAnythingConfig(
        working_dir="./rag_storage",
        parser="mineru",
        enable_image_processing=True,
    )

    # Initialize with non-streaming LLM for indexing
    rag = RAGAnything(
        config=config,
        llm_model_func=create_llm_func(api_key, stream=False),
        embedding_func=EmbeddingFunc(
            embedding_dim=1536,
            max_token_size=8192,
            func=lambda texts: openai_embed(texts, api_key=api_key),
        ),
    )

    await rag.lightrag.initialize_storages()
    yield


app = FastAPI(lifespan=lifespan)


async def stream_query(query: str, mode: str) -> AsyncGenerator[str, None]:
    """Stream RAG query results using SSE format"""
    api_key = os.getenv("OPENAI_API_KEY")

    # Get context from RAG first (non-streaming)
    context = await rag.lightrag.aquery(
        query, param={"mode": mode, "only_need_context": True}
    )

    # Now stream the LLM response with context
    system_prompt = f"""Answer the question based on the following context:

{context}

If the context doesn't contain relevant information, say so.
"""

    # Create streaming LLM call directly
    response = await openai_complete_if_cache(
        "gpt-4o-mini",
        query,
        system_prompt=system_prompt,
        api_key=api_key,
        stream=True,
        stream_options={"include_usage": True},
    )

    # response is an async generator when streaming
    async for chunk in response:
        yield f"data: {chunk}\n\n"

    yield "data: [DONE]\n\n"


@app.post("/query")
async def query(request: QueryRequest):
    if not rag:
        raise HTTPException(status_code=503, detail="RAG not initialized")

    if request.stream:
        return StreamingResponse(
            stream_query(request.query, request.mode),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            },
        )

    # Non-streaming
    result = await rag.aquery(request.query, mode=request.mode)
    return {"result": result}


@app.get("/health")
async def health():
    return {"status": "ok", "rag_initialized": rag is not None}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

# streaming_server.py
import sys

sys.path.insert(0, "/home/mujahid/PycharmProjects/rag-anything-test/raganything-source")
sys.path.insert(
    0,
    "/home/mujahid/PycharmProjects/rag-anything-test/venv/lib/python3.11/site-packages",
)
import os
import asyncio
from typing import AsyncGenerator, List, Optional
from contextlib import asynccontextmanager
from lightrag import LightRAG, QueryParam

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
import os
from transformers import AutoTokenizer, AutoModel
from lightrag.llm.hf import hf_embed
from opensearchpy import OpenSearch

class QueryRequest(BaseModel):
    query: str
    mode: str = "hybrid"
    stream: bool = False
    include_references: bool = True
    include_chunk_content: bool = True
rag: RAGAnything = None


# nest_asyncio.apply()
api_key = os.getenv("OPENAI_API_KEY")
WORKING_DIR = "./rag_storage"
def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    return openai_complete_if_cache(
            "gpt-4o-mini",
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
            # base_url=base_url,
            **kwargs,
        )
def initialize_rag():

    embedding_func = EmbeddingFunc(
        embedding_dim=1024,
        max_token_size=8192,
        func=lambda texts: hf_embed(
            texts,
            tokenizer=AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5"),
            embed_model=AutoModel.from_pretrained("BAAI/bge-large-en-v1.5"),
        ),
    )
    lightrag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,        
        embedding_func=embedding_func
        )

    lightrag.initialize_storages()  # Auto-initializes pipeline_status
    return lightrag

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
    global lightrag
    # api_key = os.getenv("OPENAI_API_KEY")

    # config = RAGAnythingConfig(
    #     working_dir="./rag_storage",
    #     parser="mineru",
    #     enable_image_processing=True,
    # )
    embedding_func = EmbeddingFunc(
        embedding_dim=1024,
        max_token_size=8192,
        func=lambda texts: hf_embed(
            texts,
            tokenizer=AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5"),
            embed_model=AutoModel.from_pretrained("BAAI/bge-large-en-v1.5"),
        ),
    )
    lightrag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,        
        embedding_func=embedding_func
        )

    await lightrag.initialize_storages()
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

class ReferenceItem(BaseModel):
    """A single reference item in query responses."""

    reference_id: str = Field(description="Unique reference identifier")
    file_path: str = Field(description="Path to the source file")
    content: Optional[List[str]] = Field(
        default=None,
        description="List of chunk contents from this file (only present when include_chunk_content=True)",
    )


class QueryResponse(BaseModel):
    response: str = Field(
        description="The generated response",
    )
    references: Optional[List[ReferenceItem]] = Field(
        default=None,
        description="Reference list (Disabled when include_references=False, /query/data always includes references.)",
    )

lightrag = initialize_rag()

@app.post("/query")
async def query(request: QueryRequest):
    try:
        # param = request.to_query_params(
        #      False
        #  )  # Ensure stream=False for non-streaming endpoint
        # Force stream=False for /query endpoint regardless of include_references setting
        
        # param.stream = False
         # Unified approach: always use aquery_llm for both cases
        # result = await lightrag.aquery_llm(request.query, param=param)
        result = await lightrag.aquery_llm(request.query)

            # Extract LLM response and references from unified result
        llm_response = result.get("llm_response", {})
        data = result.get("data", {})
        references = data.get("references", [])

            # Get the non-streaming response content
        response_content = llm_response.get("content", "")
        if not response_content:
            response_content = "No relevant context found for the query."

            # Enrich references with chunk content if requested
        if request.include_references and request.include_chunk_content:
            chunks = data.get("chunks", [])
                # Create a mapping from reference_id to chunk content
            ref_id_to_content = {}
            for chunk in chunks:
                ref_id = chunk.get("reference_id", "")
                content = chunk.get("content", "")
                if ref_id and content:
                        # Collect chunk content; join later to avoid quadratic string concatenation
                    ref_id_to_content.setdefault(ref_id, []).append(content)

                # Add content to references
            enriched_references = []
            for ref in references:
                ref_copy = ref.copy()
                ref_id = ref.get("reference_id", "")
                if ref_id in ref_id_to_content:
                        # Keep content as a list of chunks (one file may have multiple chunks)
                    ref_copy["content"] = ref_id_to_content[ref_id]
                enriched_references.append(ref_copy)
            references = enriched_references

            # Return response with or without references based on request
        if request.include_references:
            return QueryResponse(response=response_content, references=references)
        else:
            return QueryResponse(response=response_content, references=None)
    except Exception as e:
        # print(f"Error processing query: {str(e)}", exc_info=True)
        print(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok", "rag_initialized": rag is not None}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

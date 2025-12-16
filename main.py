import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal
from lightrag.utils import EmbeddingFunc
import os
from transformers import AutoTokenizer, AutoModel
from lightrag.llm.hf import hf_embed

# Import LightRAG components
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc
# from lightrag.embeddings import openai_embed # Example embedding function

# --- 1. Define Pydantic Models for API ---
class QueryRequest(BaseModel):
    """Request model for the RAG query endpoint."""
    query: str
    mode: Literal['naive', 'local', 'global', 'hybrid', 'mix'] = 'hybrid'
    only_need_context: bool = False

class RAGResponse(BaseModel):
    """Response model for the RAG query endpoint."""
    answer: str
    context: str

# --- 2. Global Variables and Initialization ---
app = FastAPI(title="LightRAG FastAPI Service")
rag_instance: LightRAG = None
WORKING_DIR = "./rag_storage" # Directory for LightRAG cache and knowledge graph files

# --- 3. LightRAG Initialization Function ---
api_key = os.getenv("OPENAI_API_KEY")
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
async def initialize_lightrag():
    """Initializes the LightRAG instance and loads/inserts data."""
    global rag_instance
    try:
        # Use a strong LLM function for knowledge graph extraction and generation
        llm_func = llm_model_func

        # Define an embedding function (e.g., using OpenAI's embedding model)
        embedding_func = EmbeddingFunc(
        embedding_dim=1024,
        max_token_size=8192,
        func=lambda texts: hf_embed(
            texts,
            tokenizer=AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5"),
            embed_model=AutoModel.from_pretrained("BAAI/bge-large-en-v1.5"),
        ),
    )

        rag_instance = LightRAG(
            working_dir=WORKING_DIR,
            llm_model_func=llm_func,
            embedding_func=embedding_func
        )
        
        # --- Data Insertion (Example: Run once during startup) ---
        sample_text = (
            "LightRAG is a new RAG system from HKUDS. "
            "It uses a lightweight knowledge graph (KG) for efficient retrieval. "
            "This KG is constructed by extracting entities and relations. "
            "FastAPI is a modern web framework for Python APIs."
        )
        
        # Use the asynchronous insert method (LightRAG handles KG construction here)
        print("Inserting sample data...")
        await rag_instance.insert(sample_text) 
        print("Data insertion complete.")
        
    except Exception as e:
        print(f"Failed to initialize LightRAG: {e}")
        # In a production app, you might want to raise an exception to prevent the server from starting
        
# --- 4. FastAPI Lifecycle Events ---

@app.on_event("startup")
async def startup_event():
    """Runs when the FastAPI application starts."""
    await initialize_lightrag()

@app.on_event("shutdown")
async def shutdown_event():
    """Runs when the FastAPI application shuts down."""
    if rag_instance:
        await rag_instance.cleanup()
        print("LightRAG cleanup complete.")

# --- 5. RAG Query Endpoint ---

@app.post("/query", response_model=RAGResponse)
async def query_rag(request: QueryRequest):
    """
    Handles a RAG query using the LightRAG instance.
    """
    if rag_instance is None:
        raise HTTPException(status_code=503, detail="RAG system is not initialized.")
    
    try:
        # Create the QueryParam object from the request
        query_param = QueryParam(            
            mode=request.mode,
            only_need_context=request.only_need_context
        )
        
        # Perform the asynchronous RAG query
        # LightRAG handles the retrieval (vector + KG) and generation steps
        result = await rag_instance.aquery(request.query, query_param)
        
        # Extract the final answer and the retrieved context
        answer = result.response
        # Context is typically a list of dicts/chunks; join for the API response
        context = "\n---\n".join([c.text for c in result.context])
        
        return RAGResponse(answer=answer, context=context)
        
    except Exception as e:
        print(f"Error during RAG query: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

# --- 6. Health Check Endpoint ---

@app.get("/health")
def health_check():
    return {"status": "ok", "rag_initialized": rag_instance is not None}

if __name__ == "__main__":
    # To run this file: uvicorn main:app --reload
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
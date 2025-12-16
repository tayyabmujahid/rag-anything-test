import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""


import os
import sys
sys.path.insert(0, "/home/mujahid/PycharmProjects/rag-anything-test/raganything-source")
sys.path.insert(
    0,
    "/home/mujahid/PycharmProjects/rag-anything-test/venv/lib/python3.12/site-packages",
)
import asyncio
from lightrag.utils import EmbeddingFunc, logger
from raganything import RAGAnything, RAGAnythingConfig
from lightrag import LightRAG, QueryParam

from transformers import AutoModel, AutoTokenizer

# ---- CONFIG ----
DOCS_DIR = "/home/mujahid/PycharmProjects/rag-anything-test/documents_lite"
OUTPUT_DIR = "./output"   # RAG index/output directory
WORKING_DIR = "./rag_storage"

BGE_MODEL_NAME = "BAAI/bge-m3"

# ---- Embedding function using bge-m3 ----
# (The bge-m3 model outputs 1024-dimensional embeddings.)

def bge_embed(texts):
    # NOTE: Downloading the model may take some time at first run.
    tokenizer = AutoTokenizer.from_pretrained(BGE_MODEL_NAME)
    model = AutoModel.from_pretrained(BGE_MODEL_NAME)
    model = model.to("cpu")
    # Batched inference
    import torch
    with torch.no_grad():
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return embeddings

embedding_func = EmbeddingFunc(
    embedding_dim=1024,
    max_token_size=8192,
    func=bge_embed
)

# ---- Index Documents with RAGAnything ----
async def index_documents_with_raganything():
    config = RAGAnythingConfig(
        working_dir=WORKING_DIR,
        enable_image_processing=False,
        enable_table_processing=False,
        enable_equation_processing=False,
    )

    rag = RAGAnything(
        config=config,
        embedding_func=embedding_func,
        # For local embedding only -- don't need LLM here, so provide dummy func
        llm_model_func=lambda prompt, **kwargs: "",
        vision_model_func=None,
    )

    # Recursively find all files in DOCS_DIR
    file_list = []
    for root, dirs, files in os.walk(DOCS_DIR):
        for file in files:
            if file.lower().endswith(('.txt', '.md', '.pdf', '.docx')):
                file_list.append(os.path.join(root, file))
    print(f"Found {len(file_list)} files to index.")
    
    # Index each file
    for file_path in file_list:
        print(f"Indexing: {file_path}")
        await rag.process_document_complete(
            file_path=file_path,
            output_dir=OUTPUT_DIR,
            parse_method="auto"
        )

# ---- Query using LightRAG ----
def query_light_rag():
    # Important: Use the same working_dir as for indexing!
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=embedding_func,
        llm_model_func=lambda prompt, **kwargs: "Dummy answer (LLM not used here)",
        llm_model_name=None,
    )
    rag.initialize_storages()  # If not already initialized

    question = "What are the main topics in these documents?"

    print("\n--- Query Results (hybrid) ---")
    result = rag.query(question, param=QueryParam(mode="hybrid"))
    print(result)

# ---- MAIN ----
def main():
    # Step 1: Index the document directory
    asyncio.run(index_documents_with_raganything())

    # Step 2: Query the store using LightRAG
    query_light_rag()

if __name__ == "__main__":
    main()
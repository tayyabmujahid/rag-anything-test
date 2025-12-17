import os
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys

sys.path.insert(0, "/home/mujahid/PycharmProjects/rag-anything-test/raganything-source")
sys.path.insert(
    0,
    "/home/mujahid/PycharmProjects/rag-anything-test/venv/lib/python3.12/site-packages",
)

from lightrag import LightRAG, QueryParam
from lightrag.llm.hf import hf_model_complete, hf_embed
from lightrag.utils import EmbeddingFunc
from transformers import AutoModel, AutoTokenizer
from lightrag.llm.openai import gpt_4o_mini_complete,openai_complete_if_cache

import asyncio
import nest_asyncio

nest_asyncio.apply()
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
async def initialize_rag():

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

    await lightrag.initialize_storages()  # Auto-initializes pipeline_status
    return lightrag

def main():
    lightrag = asyncio.run(initialize_rag())
    text_result = lightrag.query("how can i create an invoice here?",param=QueryParam(mode="hybrid"))
    print(text_result)

if __name__ == "__main__":
    main()

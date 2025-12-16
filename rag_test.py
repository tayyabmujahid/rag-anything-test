import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys

sys.path.insert(0, "/home/mujahid/PycharmProjects/rag-anything-test/raganything-source")
sys.path.insert(
    0,
    "/home/mujahid/PycharmProjects/rag-anything-test/venv/lib/python3.12/site-packages",
)


import asyncio
# from __editable___raganything_1_2_8_finder import RAGAnything, RAGAnythingConfig
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
import os
from transformers import AutoTokenizer, AutoModel
from lightrag.llm.hf import hf_embed
from opensearchpy import OpenSearch
client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    http_auth=("admin", "admin"),
    use_ssl=False,
)

# # ---- VectorStore Adapter ----
# vector_store = OpenSearchVectorStore(
#     client=client,
#     index_name="rag_index",
#     vector_field="embeddings",     # your knn_vector field
#     text_field="content",          # main text field
#     metadata_field="metadata",     # nested metadata field
# )

async def main():
    # Set up API configuration
    api_key = os.getenv("OPENAI_API_KEY")
    # base_url = "your-base-url"  # Optional

    # Create RAGAnything configuration
    config = RAGAnythingConfig(
        working_dir="./rag_storage",
        parser="mineru",  # Parser selection: mineru or docling
        parse_method="auto",  # Parse method: auto, ocr, or txt
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    # Define LLM model function
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

    # Define vision model function for image processing
    def vision_model_func(
        prompt,
        system_prompt=None,
        history_messages=[],
        image_data=None,
        messages=None,
        **kwargs,
    ):
        # If messages format is provided (for multimodal VLM enhanced query), use it directly
        if messages:
            return openai_complete_if_cache(
                "gpt-4o",
                "",
                system_prompt=None,
                history_messages=[],
                messages=messages,
                api_key=api_key,
                # base_url=base_url,
                **kwargs,
            )
        # Traditional single image format
        elif image_data:
            return openai_complete_if_cache(
                "gpt-4o",
                "",
                system_prompt=None,
                history_messages=[],
                messages=[
                    (
                        {"role": "system", "content": system_prompt}
                        if system_prompt
                        else None
                    ),
                    (
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_data}"
                                    },
                                },
                            ],
                        }
                        if image_data
                        else {"role": "user", "content": prompt}
                    ),
                ],
                api_key=api_key,
                # base_url=base_url,
                **kwargs,
            )
        # Pure text format
        else:
            return llm_model_func(prompt, system_prompt, history_messages, **kwargs)

    # Define embedding function
    embedding_func = EmbeddingFunc(
        embedding_dim=1024,
        max_token_size=8192,
        func=lambda texts: hf_embed(
            texts,
            tokenizer=AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5"),
            embed_model=AutoModel.from_pretrained("BAAI/bge-large-en-v1.5"),
        ),
    )

    # Initialize RAGAnything
    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
    )
    # await rag.process_document_complete(
    #    file_path="/home/mujahid/PycharmProjects/rag-anything-test/documents/business_devices.pdf",
    #      output_dir="./output",
    #     parse_method="auto",
    # )
    for file in os.listdir("/home/mujahid/PycharmProjects/rag-anything-test/documents/")[:2]: # Process a document
        print(f"Processing {file}")
        if file.endswith(".pdf"):
            await rag.process_document_complete(
                file_path=f"/home/mujahid/PycharmProjects/rag-anything-test/documents/{file}",
                    output_dir="./output",
                    parse_method="auto",
                    display_stats=True,
                    
                )
        else:
            print(f"Skipping {file}")
        print(f"Processed {file}")

    # Query the processed content
    # Pure text query - for basic knowledge base search
    text_result = await rag.aquery(
        "How can i create an invoice here?", mode="hybrid"
    )
    print("Text query result:", text_result)

    # Multimodal query with specific multimodal content
    multimodal_result = await rag.aquery_with_multimodal(
        "Explain to create an invoice here?",
        # multimodal_content=[
        #     {
        #         "type": "equation",
        #         "latex": "P(d|q) = \\frac{P(q|d) \\cdot P(d)}{P(q)}",
        #         "equation_caption": "Document relevance probability",
        #     }
        # ],
        mode="hybrid",
    )
    print("Multimodal query result:", multimodal_result)


if __name__ == "__main__":
    asyncio.run(main())

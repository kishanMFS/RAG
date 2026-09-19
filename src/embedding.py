import os
from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from src.data_loader import load_all_documents
# from langchain_openai import OpenAIEmbeddings

class EmbeddingPipeline:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 2. Verify that your system detects the Intel GPU
        if torch.cuda.is_available():
            target_device = "cuda"
            # NVIDIA GPUs are identified via standard CUDA properties
            gpu_name = torch.cuda.get_device_name(0)
            print(f"🚀 Success! Targeting NVIDIA GPU: {gpu_name}")
            
            # Modern NVIDIA cards (RTX 30-series/Ampere or newer) support bfloat16, 
            # while older cards (GTX 10-series/Pascal or Turing) run best on float16.
            if torch.cuda.is_bf16_supported():
                model_dtype = torch.bfloat16
                print("💡 Using bfloat16 precision.")
            else:
                model_dtype = torch.float16
                print("💡 Using float16 precision.")
        else:
            target_device = "cpu"
            print("⚠️ NVIDIA GPU not detected. Falling back to CPU/RAM.")
            model_dtype = torch.float32

        model_name = "Qwen/Qwen3-Embedding-0.6B"
        open_api_key = os.getenv("OPENAI_API_KEY")
        
        self.model = SentenceTransformer(model_name,
                                        device=target_device,
                                        model_kwargs={"torch_dtype": model_dtype})
        # 2. Initialize the model with text-embedding-3-small
        # self.model = OpenAIEmbeddings(model=model_name, api_key=open_api_key)
        print(f"[INFO] Loaded embedding model: {model_name}")

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks

    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        texts = [chunk.page_content for chunk in chunks]
        print(f"[INFO] Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        # embeddings = np.array(self.model.embed_documents(texts), dtype='float32')
        print(f"[INFO] Embeddings shape: {embeddings.shape}")
        return embeddings

# Example usage
if __name__ == "__main__":
    
    docs = load_all_documents("data")
    emb_pipe = EmbeddingPipeline()
    chunks = emb_pipe.chunk_documents(docs)
    embeddings = emb_pipe.embed_chunks(chunks)
    print("[INFO] Example embedding:", embeddings[0] if len(embeddings) > 0 else None)
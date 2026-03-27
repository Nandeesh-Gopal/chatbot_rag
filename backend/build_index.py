import os
import pickle
from pathlib import Path

import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


BACKEND_DIR = Path(__file__).parent
REPO_ROOT = BACKEND_DIR.parent


def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None or val.strip() == "":
        return default
    try:
        return int(val)
    except ValueError as e:
        raise SystemExit(f"{name} must be an int (got {val!r})") from e


def main() -> None:
    embedding_model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    data_path = Path(os.getenv("DATA_PATH", str(REPO_ROOT / "data" / "docs.txt")))
    index_path = Path(os.getenv("INDEX_PATH", str(BACKEND_DIR / "vector.index")))
    chunks_path = Path(os.getenv("CHUNKS_PATH", str(BACKEND_DIR / "chunks.pkl")))
    chunk_size = _env_int("CHUNK_SIZE", 1000)
    chunk_overlap = _env_int("CHUNK_OVERLAP", 200)

    if not data_path.exists():
        raise SystemExit(
            f"DATA_PATH not found: {data_path}\n"
            f"Set DATA_PATH to a .txt file (default is {REPO_ROOT / 'data' / 'docs.txt'})."
        )

    print(f"Loading embedding model: {embedding_model_name}")
    model = SentenceTransformer(embedding_model_name)

    print(f"Reading data: {data_path}")
    text = data_path.read_text(encoding="utf-8")

    print(f"Splitting text (chunk_size={chunk_size}, chunk_overlap={chunk_overlap})")
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    if not chunks:
        raise SystemExit("No chunks produced; check your input document.")
    print(f"Total chunks: {len(chunks)}")

    print("Creating embeddings...")
    embeddings = model.encode(chunks).astype("float32")

    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

    print("Index built successfully!")
    print(f"Wrote: {index_path}")
    print(f"Wrote: {chunks_path}")


if __name__ == "__main__":
    main()
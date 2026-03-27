import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load embedding model
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load document
with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_text(text)
print(f"Total chunks: {len(chunks)}")

# Create embeddings
print("Creating embeddings...")
embeddings = model.encode(chunks)

# Convert to float32
embeddings = embeddings.astype("float32")

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save index
faiss.write_index(index, "vector.index")

# Save chunks separately
with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("Index built successfully!")
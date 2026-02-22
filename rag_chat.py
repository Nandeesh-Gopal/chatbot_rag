import faiss
import pickle
import numpy as np
import ollama
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index
index = faiss.read_index("vector.index")

# Load stored chunks
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

print("RAG system ready! Type 'exit' to quit.\n")

while True:
    query = input("Ask a question: ")

    if query.lower() == "exit":
        break

    # Convert query to embedding
    query_embedding = model.encode([query]).astype("float32")

    # Search top 3 similar chunks
    k = 3
    distances, indices = index.search(query_embedding, k)

    # Retrieve context
    context = "\n\n".join([chunks[i] for i in indices[0]])

    # Send to Ollama
    response = ollama.chat(
        model="llama3",
        messages=[
            {"role": "system", "content": "Answer only from the provided context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"}
        ]
    )

    print("\nAnswer:\n")
    print(response["message"]["content"])
    print("\n" + "-"*50 + "\n")
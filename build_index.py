import os
#used for defining file path
import faiss
#facebook ai similarity search
#used for store embeddings
import pickle
#used to save the python objects into a file(serialization)
from sentence_transformers import SentenceTransformer
#text to emmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
#chunk splitter librarary

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
#384 dimensional vectors

# Load document
with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200#Prevents loss of context between chunks.

)

chunks = text_splitter.split_text(text)
#print(chunks)
print(f"Total chunks: {len(chunks)}")

# Create embeddings
print("Creating embeddings...")
embeddings = model.encode(chunks)
#If you have 25 chunks → you get 25 vectors.
# Convert to float32
embeddings = embeddings.astype("float32")
#FAISS requires embeddings in float32 format.
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
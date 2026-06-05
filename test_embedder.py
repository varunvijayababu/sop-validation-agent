from sentence_transformers import SentenceTransformer

print("Loading...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Loaded")
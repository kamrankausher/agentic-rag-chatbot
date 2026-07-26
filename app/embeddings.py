from langchain_community.embeddings import HuggingFaceEmbeddings


embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

text = "Agentic AI enables autonomous decision making."

vector = embedding_model.embed_query(text)

print("Vector Length :", len(vector))
print("First 10 Values:\n")
print(vector[:10])
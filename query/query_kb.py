from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Load the same embeddings model used during ingestion
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load FAISS index
vectorstore = FAISS.load_local(
    "embeddings/faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

print("✅ Knowledge base loaded")

# Query loop
while True:
    query = input("\nAsk a question (or type 'exit'): ")
    if query.lower() == "exit":
        break

    results = vectorstore.similarity_search(query, k=3)

    print("\n🔍 Relevant information:")
    for i, doc in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(doc.page_content)

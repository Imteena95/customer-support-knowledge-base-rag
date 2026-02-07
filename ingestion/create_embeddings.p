import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSpiltter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load documents
ticket_loader = TextLoader("data/tickets/tickets.txt")
doc_loader = TextLoader("data/docs/help_docs.txt")

tickets = ticket_loader.load()
docs = doc_loader.load()

documents = tickets + docs

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

# Create embeddings
embeddings = OpenAIEmbeddings()

# Store in FAISS
vectorstore = FAISS.from_documents(chunks, embeddings)

vectorstore.save_local("embeddings/faiss_index")

print("✅ Embeddings created and saved successfully")

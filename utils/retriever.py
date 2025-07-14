from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema.document import Document
import os

embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=".chroma_store", embedding_function=embedding_function)

def add_to_chroma(texts, ids):
    docs = [Document(page_content=t, metadata={"id": i}) for t, i in zip(texts, ids)]
    vectorstore.add_documents(docs)

def query_chroma(query, n_results=3):
    docs = vectorstore.similarity_search(query, k=n_results)
    return [doc.page_content for doc in docs]

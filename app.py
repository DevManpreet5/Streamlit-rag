import streamlit as st
import os
import uuid
from utils.chunking import chunk_text
from utils.embedding import embed_texts
from utils.retriever import add_to_chroma, query_chroma
from utils.prompt import format_prompt
from utils.completion import complete

st.title("Streamlit RAG with Chroma and LangChain")

# Load and chunk documents
if st.button("Index Documents"):
    st.write("Indexing...")
    data_dir = "data"
    raw_texts = []
    for filename in os.listdir(data_dir):
        with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as f:
            raw_texts.append(f.read())

    chunks = []
    ids = []
    for i, text in enumerate(raw_texts):
        split_chunks = chunk_text(text)
        chunks.extend(split_chunks)
        ids.extend([str(uuid.uuid4()) for _ in split_chunks])

    add_to_chroma(chunks, ids)
    st.success("Documents indexed successfully.")

# Query interface
query = st.text_input("Ask a question")
if query:
    contexts = query_chroma(query, n_results=3)
    prompt = format_prompt(query, contexts)
    answer = complete(prompt)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Top Contexts")
    for idx, context in enumerate(contexts):
        st.markdown(f"**Context {idx+1}:**\n{context}")
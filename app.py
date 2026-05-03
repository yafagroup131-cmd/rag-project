import os
from dotenv import load_dotenv

load_dotenv("env.txt")

# 👇 اختبار مؤقت
print(os.getenv("GOOGLE_API_KEY"))

import streamlit as st

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


# ===== UI =====
st.title("🤖 Retriva RAG")


st.markdown("""
<style>
body {
    font-family: 'DM Sans', sans-serif;
    background-color: #E8F5E9;  /* Mint Green فاتح */
    color: #1B5E20;  /* أخضر غامق للنص */
}

h1 {
    text-align: center;
    color: #2E7D32;
}

.stButton>button {
    background-color: #66BB6A;  /* زرار أخضر */
    color: white;
    border-radius: 10px;
}

.stTextInput>div>div>input {
    background-color: #FFFFFF;
    color: black;
    border-radius: 8px;
}

</style>
""", unsafe_allow_html=True)


# ===== Embeddings =====
embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    encode_kwargs={'normalize_embeddings': True}
)


# ===== Vector DB =====
vectorstore = Chroma(
    persist_directory="./vector_DB",
    embedding_function=embedding
)

retriever = vectorstore.as_retriever()

import os

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    google_api_key=st.secrets["GOOGLE_API_KEY"]
)


# ===== Prompt =====
prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the context below.

<context>
{context}
</context>

Question: {input}
""")


# ===== Helper function =====
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def format_apa(doc):
    source = doc.metadata.get("source", "Unknown")
    page = doc.metadata.get("page", "")
    doi = doc.metadata.get("doi", "")

    # نجيب اسم الملف بس
    title = source.split("\\")[-1].replace(".pdf", "")

    citation = f"{title}. (n.d.)."

    if doi:
        citation += f" https://doi.org/{doi}"

    if page != "":
        citation += f" (p. {page})"

    return citation

# ===== RAG PIPELINE (LCEL) =====
rag_chain = (
    {
        "context": lambda x: format_docs(retriever.invoke(x["input"])),
        "input": lambda x: x["input"]
    }
    | prompt
    | llm
)

# ===== Input =====
user_input = st.text_input("اكتب سؤالك هنا:")


# ===== Run =====
if st.button("اسأل"):
    if user_input:

        response = rag_chain.invoke({
            "input": user_input
        })

        st.subheader("📌 Answer")
        st.write(response.content)

        st.subheader("📚 References (APA)")

        docs = retriever.invoke(user_input)

        for i, doc in enumerate(docs):
            citation = format_apa(doc)
            st.write(f"{i+1}. {citation}")
            

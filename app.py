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

import base64

logo = base64.b64encode(open("assets/logo.png", "rb").read()).decode()

st.markdown(f"""
<div style="
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    z-index: 999;
    background-color: #81C784;
    padding: 10px;
    display: flex;
    justify-content: center;
">
    <img src="data:image/png;base64,{logo}" width="200">
</div>

<style>

/* ===== الخلفية ===== */
[data-testid="stAppViewContainer"] {{
    background-color: #E8F5E9;
}}

/* 👇 لازم نزوّق المساحة تحت الهيدر */
.block-container {{
    padding-top: 100px;
}}

/* ===== الخط ===== */
html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif;
}}

/* ===== إخفاء Streamlit UI ===== */
#MainMenu, footer, header {{
    visibility: hidden;
}}

/* ===== رسائل المستخدم ===== */
[data-testid="stChatMessage"][data-testid*="user"] {{
    background-color: #A5D6A7;
    border-radius: 15px;
    padding: 10px;
}}

/* ===== رسائل البوت ===== */
[data-testid="stChatMessage"][data-testid*="assistant"] {{
    background-color: white;
    border-radius: 15px;
    padding: 10px;
}}

/* ===== input ===== */
[data-testid="stChatInput"] {{
    border-radius: 10px;
}}

/* ===== زرار الإرسال (السهم) ===== */
[data-testid="stChatInput"] button {
    background-color: #81C784 !important;
}

[data-testid="stChatInput"] button:hover {
    background-color: #66BB6A !important;
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
        

# ===== Chat Memory =====
if "messages" not in st.session_state:
    st.session_state.messages = []

# ===== عرض الرسائل القديمة =====
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# ===== Input =====
user_input = st.chat_input("Ask your question...")

# ===== Run =====
if user_input:

    # 🧑 رسالة المستخدم
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.write(user_input)

    # 🤖 تشغيل RAG
    with st.spinner("Thinking..."):
        response = rag_chain.invoke({
            "input": user_input
        })

    answer = response.content

    # 🤖 رسالة البوت
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    with st.chat_message("assistant"):
        st.write(answer)

        # 📚 المصادر (Dropdown)
        with st.expander("📚 View References"):
            docs = retriever.invoke(user_input)

            for i, doc in enumerate(docs):
                citation = format_apa(doc)
                st.write(f"{i+1}. {citation}")

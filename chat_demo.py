# drug_chatbot_app.py

import os
import streamlit as st
import pandas as pd
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


# 🖼️ Streamlit setup
st.set_page_config(page_title="💊 Drug Info Chatbot (Custom RAG)", layout="wide")
st.title("💊 Drug Information Chatbot (with Custom RAG Pipeline)")

# 📄 Load the drug CSV file
try:
    df = pd.read_csv("DrugData.csv")
    st.success("Drug database loaded successfully.")
except Exception as e:
    st.error(f"Error loading DrugData.csv: {e}")
    st.stop()

# 🧾 Convert each row into a LangChain Document
def row_to_doc(row):
    return Document(page_content="\n".join([
        f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col])
    ]))

docs = [row_to_doc(row) for _, row in df.iterrows()]

# 🧠 Create vector store
embedding = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embedding)
retriever = vectorstore.as_retriever()

# 🤖 Define LLM and custom prompt
llm = ChatOpenAI(model_name="gpt-4", temperature=0)

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant trained on pharmaceutical drug data.
Only use the context below to answer the question.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question:
{question}
"""
)

# 🔗 Define RAG pipeline with custom chaining
rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)

# 💬 Streamlit input and output
query = st.text_input("Ask a question about any drug:")

if query:
    with st.spinner("Searching the drug database..."):
        response = rag_chain.invoke(query)
    st.markdown("### 🧠 Answer")
    st.success(response.content if hasattr(response, "content") else response)
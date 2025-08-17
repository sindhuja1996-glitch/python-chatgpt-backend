from langchain_community.document_loaders import TextLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import streamlit as st

from langchain_groq import ChatGroq
import os

## load documents
loader = DirectoryLoader("./documents")
documents = loader.load()
## split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)
## create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",model_kwargs={"device": "cpu"} )
## create vector store  
vector_store = FAISS.from_documents(texts, embeddings)
## create retriever
retriever = vector_store.as_retriever()
## create prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", """Answer the question based on the context provided below which has to exaxtly matached with question if question not matched with context say not availabel. 
If you don't know the answer, just say you don't know.
<context> {context}</context>
Question: {input}""")
])
##
#  create retrival chain chain
document_chain = create_stuff_documents_chain( 
    llm=ChatGroq(api_key="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",model_name="llama-3.3-70b-versatile"),
    prompt=prompt
)
chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=document_chain,
)

## define a function to answer questions
def answer_question(question):
    response = chain.invoke({"input": question})
    return response

## Streamlit app
st.title("Tax Streamlit App")
st.write("Ask your tax-related questions below:")
question = st.text_input("Enter your question:")
if question:
    with st.spinner("Processing..."):
        answer = answer_question(question)
    st.write("Response:", answer['answer'])



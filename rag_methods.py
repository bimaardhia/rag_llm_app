import os
import dotenv
from time import time
import streamlit as st
import random

from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders import (
    WebBaseLoader, 
    PyPDFLoader, 
    Docx2txtLoader,
)
# pip install docx2txt, pypdf
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain, LLMChain
from langchain.chains.combine_documents import create_stuff_documents_chain

dotenv.load_dotenv()

os.environ["USER_AGENT"] = "myagent"
DB_DOCS_LIMIT = 10

# Function to stream the response of the LLM 
def stream_llm_response(llm_stream, messages):
    response_message = ""

    for chunk in llm_stream.stream(messages):
        response_message += chunk.content
        yield chunk

    st.session_state.messages.append({"role": "assistant", "content": response_message})


# --- Indexing Phase ---

def load_doc_to_db():
    # Use loader according to doc type
    if "rag_docs" in st.session_state and st.session_state.rag_docs:
        docs = [] 
        for doc_file in st.session_state.rag_docs:
            if doc_file.name not in st.session_state.rag_sources:
                if len(st.session_state.rag_sources) < DB_DOCS_LIMIT:
                    os.makedirs("source_files", exist_ok=True)
                    file_path = f"./source_files/{doc_file.name}"
                    with open(file_path, "wb") as file:
                        file.write(doc_file.read())

                    try:
                        if doc_file.type == "application/pdf":
                            loader = PyPDFLoader(file_path)
                        elif doc_file.name.endswith(".docx"):
                            loader = Docx2txtLoader(file_path)
                        elif doc_file.type in ["text/plain", "text/markdown"]:
                            loader = TextLoader(file_path)
                        else:
                            st.warning(f"Document type {doc_file.type} not supported.")
                            continue

                        docs.extend(loader.load())
                        st.session_state.rag_sources.append(doc_file.name)

                    except Exception as e:
                        st.toast(f"Error loading document {doc_file.name}: {e}", icon="⚠️")
                        print(f"Error loading document {doc_file.name}: {e}")
                    
                    finally:
                        os.remove(file_path)

                else:
                    st.error(F"Maximum number of documents reached ({DB_DOCS_LIMIT}).")

        if docs:
            _split_and_load_docs(docs)
            st.toast(f"Document *{str([doc_file.name for doc_file in st.session_state.rag_docs])[1:-1]}* loaded successfully.", icon="✅")


def load_default_file(default_file_path):
    if os.path.exists(default_file_path):
        file_name = os.path.basename(default_file_path)
        if file_name not in st.session_state.rag_sources:
            docs = []
            try:
                if default_file_path.endswith(".pdf"):
                    loader = PyPDFLoader(default_file_path)
                elif default_file_path.endswith(".docx"):
                    loader = Docx2txtLoader(default_file_path)
                elif default_file_path.endswith((".txt", ".md")):
                    loader = TextLoader(default_file_path)
                else:
                    st.warning(f"Default document type not supported: {file_name}")
                    return

                docs.extend(loader.load())
                st.session_state.rag_sources.append(file_name)

                if docs:
                    _split_and_load_docs(docs)
                    st.toast(f"Default document *{file_name}* loaded successfully.", icon="✅")
            except Exception as e:
                st.toast(f"Error loading default document {file_name}: {e}", icon="⚠️")
                print(f"Error loading default document {file_name}: {e}")
    else:
        st.error("Default file not found. Please check the file path.")

def load_url_to_db():
    if "rag_url" in st.session_state and st.session_state.rag_url:
        url = st.session_state.rag_url
        docs = []
        if url not in st.session_state.rag_sources:
            if len(st.session_state.rag_sources) < 10:
                try:
                    loader = WebBaseLoader(url)
                    docs.extend(loader.load())
                    st.session_state.rag_sources.append(url)

                except Exception as e:
                    st.error(f"Error loading document from {url}: {e}")

                if docs:
                    _split_and_load_docs(docs)
                    st.toast(f"Document from URL *{url}* loaded successfully.", icon="✅")

            else:
                st.error("Maximum number of documents reached (10).")


def initialize_vector_db(docs):
    embedding = OpenAIEmbeddings(api_key=st.session_state.openai_api_key)
    
    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        collection_name=f"{str(time()).replace('.', '')[:14]}_" + st.session_state['session_id'],
    )

    # We need to manage the number of collections that we have in memory, we will keep the last 20
    chroma_client = vector_db._client
    collection_names = sorted([collection.name for collection in chroma_client.list_collections()])
    print("Number of collections:", len(collection_names))
    while len(collection_names) > 20:
        chroma_client.delete_collection(collection_names[0])
        collection_names.pop(0)

    return vector_db


def _split_and_load_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=1000,
    )

    document_chunks = text_splitter.split_documents(docs)

    if "vector_db" not in st.session_state:
        st.session_state.vector_db = initialize_vector_db(docs)
    else:
        st.session_state.vector_db.add_documents(document_chunks)


# --- Retrieval Augmented Generation (RAG) Phase ---

def _get_context_retriever_chain(vector_db, llm):
    retriever = vector_db.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
        ("user", "Based on the conversation above, create a search query to find information relevant to the conversation, focusing on the latest messages."),
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain


def get_conversational_rag_chain(llm):
    retriever_chain = _get_context_retriever_chain(st.session_state.vector_db, llm)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
        """You are a teacher helping your student learn. When responding to the user's question, guide the student through the steps needed to solve the problem, explaining each part clearly and systematically. Do not give the final answer. Use the provided context to support the explanation, but do not explicitly mention the context.

        Context:
        {context}"""),
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
    ])

    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


def stream_llm_rag_response(llm_stream, messages):
    conversation_rag_chain = get_conversational_rag_chain(llm_stream)
    response_message = ""
    for chunk in conversation_rag_chain.pick("answer").stream({"messages": messages[:-1], "input": messages[-1].content}):
        response_message += chunk
        yield chunk

    st.session_state.messages.append({"role": "assistant", "content": response_message})


# lat sol 


def get_random_chunk(vector_db):
    """
    Retrieve a random chunk from VectorDB.
    """
    retriever = vector_db.as_retriever()
    docs = retriever.get_relevant_documents("random")  # Random query untuk mengambil dokumen
    return random.choice(docs).page_content if docs else None

def create_question_prompt_template():
    """
    Create a prompt template to generate multiple-choice questions.
    """
    return PromptTemplate(
        input_variables=["context"],
        template="""
As a teacher, your task is to create a multiple-choice question with four options based on the following context. Ensure the explanation for the correct answer is accurate and clearly explains why each of the other options is incorrect, one by one.

Based on the following context, create a multiple-choice question with four options.
        Include:
        - The question
        - Four answer options
        - The correct answer
        - An explanation for the correct answer

        Context:
        {context}

        Output format:

        **Question:**
        ...

        | **Option** | **Answer** |
        |-------------|---------------------|
        | **A.** | ...                 |
        | **B.** | ...                 |
        | **C.** | ...                 |
        | **D.** | ...                 |

        **Correct Answer:**
        ...

        **Explanation:**
        ...
        """
    )

def generate_question_from_chunk(llm, chunk):
    """
    Use LLM to generate a multiple-choice question based on the given chunk.
    """
    if not chunk:
        return "No data available to generate a question."
    
    prompt = create_question_prompt_template()
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run({"context": chunk})
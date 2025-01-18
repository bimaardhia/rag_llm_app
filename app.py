import streamlit as st
import os
import dotenv
import uuid

# check if it's linux so it works on Streamlit Cloud
if os.name == 'posix':
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, AIMessage

from rag_methods import (
    load_doc_to_db, 
    stream_llm_response,
    stream_llm_rag_response,
)

dotenv.load_dotenv()

if "AZ_OPENAI_API_KEY" not in os.environ:
    MODELS = [
        # "openai/o1-mini",
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
        "anthropic/claude-3-5-sonnet-20240620",
    ]
else:
    MODELS = ["azure-openai/gpt-4o"]


st.set_page_config(
    page_title="RAG LLM app?", 
    page_icon="📚", 
    layout="centered", 
    initial_sidebar_state="expanded"
)


# --- Header ---
st.html("""<h2 style="text-align: center;">📚🔍 <i> Do your LLM even RAG bro? </i> 🤖💬</h2>""")


# --- Initial Setup ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "rag_sources" not in st.session_state:
    st.session_state.rag_sources = []

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there! How can I assist you today?"}
]



default_openai_api_key = openai_api_key = st.secrets["api"]["openai_api_key"]
openai_api_key = default_openai_api_key

        
# Fungsi untuk memuat CSS
def load_css():
    with open("static/styles.css", "r") as f:
        css = f"<style>{f.read()}</style>"
        st.markdown(css, unsafe_allow_html=True)

# Memuat CSS
load_css()


# --- Main Content ---
# Checking if the user has introduced the OpenAI API Key, if not, a warning is displayed
missing_openai = openai_api_key == "" or openai_api_key is None or "sk-" not in openai_api_key
if missing_openai:
    st.write("#")
    st.warning("⬅️ Please introduce an API Key to continue...")
    


else:
    # Sidebar
    with st.sidebar:
        st.divider()
        models = []
        for model in MODELS:
            if "openai" in model and not missing_openai:
                models.append(model)


        st.selectbox(
            "🤖 Select a Model", 
            options=models,
            key="model",
        )

        cols0 = st.columns(2)
        with cols0[0]:
            is_vector_db_loaded = ("vector_db" in st.session_state and st.session_state.vector_db is not None)
            st.toggle(
                "Use RAG", 
                value=is_vector_db_loaded, 
                key="use_rag", 
                disabled=not is_vector_db_loaded,
            )

        with cols0[1]:
            st.button("Clear Chat", on_click=lambda: st.session_state.messages.clear(), type="primary")

        st.header("RAG Sources:")
            
        # File upload input for RAG with documents
        st.file_uploader(
            "📄 Upload a document", 
            type=["pdf", "txt", "docx", "md"],
            accept_multiple_files=True,
            on_change=load_doc_to_db,
            key="rag_docs",
        )
        
        with st.expander(f"📚 Documents in DB ({0 if not is_vector_db_loaded else len(st.session_state.rag_sources)})"):
            st.write([] if not is_vector_db_loaded else [source for source in st.session_state.rag_sources])

    
    # Main chat app
    model_provider = st.session_state.model.split("/")[0]
    if model_provider == "openai":
        llm_stream = ChatOpenAI(
            api_key=openai_api_key,
            model_name=st.session_state.model.split("/")[-1],
            temperature=0.3,
            streaming=True,
        )

    # Apply styling for all chat messages
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]

        # Set style attributes based on role
        icon = "ai_icon.png" if role == "assistant" else "user_icon.png"
        bubble_class = "ai-bubble" if role == "assistant" else "human-bubble"
        row_class = "" if role == "assistant" else "row-reverse"

        # HTML for each message
        div = f"""
        <div class="chat-row {row_class}">
            <img class="chat-icon" src="app/static/{icon}" width=32 height=32>
            <div class="chat-bubble {bubble_class}">
                &#8203;{content}
            </div>
        </div>
        """
        st.markdown(div, unsafe_allow_html=True)

# Main logic for handling the input and output in the UI
    if prompt := st.chat_input("Your message"):
        # Immediately display the user message with custom style
        div = f"""
        <div class="chat-row row-reverse">
            <img class="chat-icon" src="app/static/user_icon.png" width=32 height=32>
            <div class="chat-bubble human-bubble">
                &#8203;{prompt}
            </div>
        </div>
        """
        st.markdown(div, unsafe_allow_html=True)

        # Add the user message to the session state
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Handle assistant's response (with custom styling)
        message_placeholder = st.empty()
        full_response = ""

        # Prepare messages for processing
        messages = [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in st.session_state.messages]

        # Get the assistant's response
        if not st.session_state.use_rag:
            assistant_response = stream_llm_response(llm_stream, messages)
        else:
            assistant_response = stream_llm_rag_response(llm_stream, messages)

        # Since the response is a generator, we need to handle the output correctly
        assistant_response_str = "".join([chunk.content if hasattr(chunk, 'content') else str(chunk) for chunk in assistant_response])

        # Render assistant's message with custom style, only display once
        assistant_div = f"""
        <div class="chat-row">
            <img class="chat-icon" src="app/static/ai_icon.png" width=32 height=32>
            <div class="chat-bubble ai-bubble">
                &#8203;{assistant_response_str}
            </div>
        </div>
        """
        st.markdown(assistant_div, unsafe_allow_html=True)

    


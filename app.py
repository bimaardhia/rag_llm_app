import streamlit as st
import os
import uuid
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="RAG LLM app?", 
    page_icon="📚", 
    layout="centered", 
    initial_sidebar_state="expanded"
)
with st.sidebar:
    selected = option_menu(
                menu_title=None,  # required
                options=["ChatBot", "Latihan Soal"],  # required
                icons=["robot", "journal-bookmark-fill"],  # optional
                menu_icon="cast",  # optional
                default_index=0,  # optional
                orientation="horizontal",
            )

# check if it's linux so it works on Streamlit Cloud
if os.name == 'posix':
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage

from rag_methods import (
    load_doc_to_db, 
    stream_llm_response,
    stream_llm_rag_response,
    get_random_chunk, 
    generate_question_from_chunk,
    load_default_file
)

default_file_path = "./docs/Kabinet Merah Putih.pdf"  # Path to your default file


if "AZ_OPENAI_API_KEY" not in os.environ:
    MODELS = [
        # "openai/o1-mini",
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
        "anthropic/claude-3-5-sonnet-20240620",
    ]
else:
    MODELS = ["azure-openai/gpt-4o"]




# --- Header ---
st.html("""<h2 style="text-align: center;">📚🔍 <i> Tutor AI </i> 🤖💬</h2>""")


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



openai_api_key = st.secrets["OPENAI_API_KEY"]
st.session_state.openai_api_key = openai_api_key

        
# Fungsi untuk memuat CSS
def load_css():
    with open("static/styles.css", "r") as f:
        css = f"<style>{f.read()}</style>"
        st.markdown(css, unsafe_allow_html=True)

# Memuat CSS
load_css()

if selected == "ChatBot":


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
            
            if st.button("Gunakan File Default: Kabinet Merah Putih"):
                load_default_file(default_file_path)
                st.rerun()
                
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

    if not is_vector_db_loaded:
        st.warning("Kamu belum upload Materi. Silakan upload dokumen terlebih dahulu.")
    else:    
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

if selected == "Latihan Soal":
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

            
            is_vector_db_loaded = ("vector_db" in st.session_state and st.session_state.vector_db is not None)
                

            st.header("RAG Sources:")
            

            if st.button("Gunakan File Default: Kabinet Merah Putih"):
                load_default_file(default_file_path)
                st.rerun()

                
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
                
            model_provider = st.session_state.model.split("/")[0]
            if model_provider == "openai":
                llm_stream = ChatOpenAI(
                    api_key=openai_api_key,
                    model_name=st.session_state.model.split("/")[-1],
                    temperature=0.3,
                    streaming=True,
                )
    
 
    
# Cek apakah VectorDB sudah dimuat
    # Cek apakah VectorDB sudah dimuat
    is_vector_db_loaded = "vector_db" in st.session_state and st.session_state.vector_db is not None
    if not is_vector_db_loaded:
        st.warning("Kamu belum upload Materi. Silakan upload dokumen terlebih dahulu.")
    else:
        # Tombol untuk menunjukkan soal yang ada di session_state
        # Tombol Generate Question
        if st.button("Generate Question"):
            chunk = get_random_chunk(st.session_state.vector_db)
            if chunk:
                # Generate question dari chunk
                question = generate_question_from_chunk(llm_stream, chunk)

                # Gantikan soal yang lama dengan soal yang baru di session_state
                st.session_state.generated_question = question  # Gantikan soal di session_state

                # Tampilkan soal yang dihasilkan
                st.markdown(question)  # Menampilkan soal dalam format Markdown
            else:
                st.warning("Tidak ada data untuk menghasilkan soal.")

        # Jika soal sudah ada di session_state, tampilkan soal yang disimpan
        elif "generated_question" in st.session_state:
            st.markdown(st.session_state.generated_question)  # Menampilkan soal yang ada di session_state



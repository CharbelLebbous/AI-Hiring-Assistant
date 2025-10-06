"""
app.py — Streamlit Frontend for the AI Hiring Assistant
------------------------------------------------------
This file defines the interactive web interface for the AI Hiring Assistant.
It allows HR professionals to:
- Upload and index resumes (PDF, DOCX, TXT)
- Delete or rebuild stored vector indexes
- Chat conversationally with the AI assistant to extract or analyze candidate information
"""

import streamlit as st
import os
import shutil
from index_builder import build_index_from_folder
from query_engine import query_assistant


# ------------------- PAGE CONFIGURATION -------------------
# Set up the Streamlit app’s title and layout.
st.set_page_config(page_title="AI Hiring Assistant", layout="wide")
st.title("🤖 AI Hiring Assistant — Conversational HR Support")


# ------------------- SIDEBAR: SETUP CONTROLS -------------------
st.sidebar.header("⚙️ Indexing & Setup")

# Input field for the resumes folder (default = ./data/resumes)
folder = st.sidebar.text_input("📁 Folder with resumes", value="./data/resumes")

# Button to build or rebuild the FAISS index from resume files
if st.sidebar.button("🔍 Build / Rebuild Index"):
    st.sidebar.info("Building index... please wait ⏳")
    build_index_from_folder(folder)  # Calls index_builder.py to process and embed resumes
    st.sidebar.success("✅ Index built successfully!")

# Button to delete the entire vector index and metadata storage
if st.sidebar.button("🗑️ Delete Stored Index & Metadata"):
    try:
        if os.path.exists("./storage"):
            shutil.rmtree("./storage")  # Delete the storage directory entirely
        st.sidebar.success("🧹 All index data deleted successfully!")
    except Exception as e:
        st.sidebar.error(f"Error deleting storage: {e}")

# Sidebar footer info
st.sidebar.markdown("---")
st.sidebar.caption("You can rebuild the index anytime after deleting.")


# ------------------- MAIN CHAT INTERFACE -------------------
st.subheader("💬 Chat with your AI Hiring Assistant")

# Example questions to guide the HR user
st.markdown("""
Ask anything about your candidates:
- *“Show me the top 3 data scientists”*  
- *“What is Charbel’s email address?”*  
- *“Who has experience in NLP?”*  
""")

# Initialize chat memory (stored in session)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat history with message bubbles
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input box at the bottom of the screen
if prompt := st.chat_input("Ask something about your candidates..."):
    # Save the user's question to session memory
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display the user's message immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate AI response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking... please wait"):
            # Send query + chat history to the AI backend
            answer, sources = query_assistant(prompt, chat_history=st.session_state.messages)

            # Display AI response
            st.markdown(answer)

            # Show unique source files used for generating the response
            if sources:
                st.markdown("### 📄 Source Files Used:")
                for src in sources:
                    st.markdown(f"- [{src['file_name']}]({src['file_path']})")
            else:
                st.markdown("📁 No sources available for this response.")

    # Save the AI's reply into chat memory for multi-turn conversation
    st.session_state.messages.append({"role": "assistant", "content": answer})

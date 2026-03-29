import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000/rag/")


# --- Sidebar ---
st.set_page_config(page_title="Hephaestus RAG Chatbot", layout="wide")

# --- Chat ---
st.title("Hephaestus RAG Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    API_URL,
                    json={
                        "query": prompt,
                    },
                    timeout=60,
                )
                response.raise_for_status()
                answer = response.json()["answer"]
            except requests.exceptions.ConnectionError:
                answer = "Could not connect to the API. Make sure the server is running."
            except Exception as e:
                answer = f"Error: {e}"

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

import json
import os
import requests
import streamlit as st
import uuid

API_URL = os.getenv("API_URL", "http://localhost:8000/rag/")
STREAM_URL = API_URL.rstrip("/") + "/stream"
# Related to feedback: define the feedback endpoint
FEEDBACK_URL = API_URL.replace("/rag/", "/submit_feedback/")


# --- Functions ---


def get_session_id():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    return st.session_state.session_id


THREAD_ID = get_session_id()


# --- Sidebar ---
st.set_page_config(page_title="Hephaestus RAG Chatbot", layout="wide")

# --- Chat ---
st.title("Hephaestus RAG Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []


# Custom function for feedback submission (related to feedback)
def submit_fb(trace_id, value, text=""):
    try:
        requests.post(
            FEEDBACK_URL,
            json={"trace_id": trace_id, "feedback_value": value, "feedback_text": text},
            timeout=5,
        )
        st.toast("Feedback received!")
    except Exception as e:
        st.error(f"Error: {e}")


for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Intuitive feedback UI based on user design
        if msg["role"] == "assistant" and "trace_id" in msg:
            fb_key = f"fb_{msg['trace_id']}_{i}"
            done_key = f"done_{fb_key}"
            recorded_key = f"recorded_{fb_key}"

            # If not yet finalized (closed or sent)
            if not st.session_state.get(done_key):
                feedback = st.feedback("thumbs", key=fb_key)

                # If a thumb has been selected
                if feedback is not None:
                    # 1. Immediately record the thumb if not already done
                    if not st.session_state.get(recorded_key):
                        submit_fb(msg["trace_id"], feedback)
                        st.session_state[recorded_key] = True

                    # 2. Show the expanded feedback form
                    st.markdown("**Want to tell us more? (Optional)**")
                    feedback_type = "positive" if feedback == 1 else "negative"
                    st.caption(
                        f"Your {feedback_type} feedback has already been recorded. You can optionally provide additional details below."
                    )

                    comment = st.text_area(
                        "Additional feedback (optional)",
                        key=f"text_area_{fb_key}",
                        height=100,
                    )

                    # 3. Footer buttons: Send and Close
                    col_send, col_close, col_spacer = st.columns([0.09, 0.09, 1.3])
                    with col_send:
                        if st.button(
                            "Send", key=f"send_full_{fb_key}", type="secondary"
                        ):
                            submit_fb(msg["trace_id"], feedback, comment)
                            st.session_state[done_key] = True
                            st.rerun()
                    with col_close:
                        if st.button("Close", key=f"close_fb_{fb_key}"):
                            st.session_state[done_key] = True
                            st.rerun()
            else:
                # Post-submission state
                st.caption("Feedback received. Thank you!")

if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        answer = ""
        trace_id = None

        try:
            with st.status("Processing...", expanded=True) as status_box:
                with requests.post(
                    STREAM_URL,
                    json={"query": prompt, "thread_id": THREAD_ID},
                    timeout=120,
                    stream=True,
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines(decode_unicode=True):
                        if not line or not line.startswith("data: "):
                            continue
                        payload = line[len("data: "):]
                        if payload == "[DONE]":
                            break
                        event = json.loads(payload)

                        if event.get("trace_id"):
                            trace_id = event["trace_id"]

                        if event["event"] == "status":
                            st.write(event["data"])
                        elif event["event"] == "tool_calls":
                            for t in event["data"]:
                                st.write(f"- {t}")
                        elif event["event"] == "answer":
                            answer = event["data"]

                status_box.update(label="Done", state="complete", expanded=False)

        except requests.exceptions.ConnectionError:
            answer = "Could not connect to the API. Make sure the server is running."
        except Exception as e:
            answer = f"Error: {e}"

        st.markdown(answer or "I could not find a relevant answer.")

        if trace_id or answer:
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "trace_id": trace_id}
            )
            st.rerun()

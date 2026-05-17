import json
import os
import requests
import streamlit as st
import uuid

API_URL = os.getenv("API_URL", "http://localhost:8000/multiagent/")
STREAM_URL = API_URL.rstrip("/") + "/stream"
FEEDBACK_URL = API_URL.replace("/multiagent/", "/submit_feedback/")


WELCOME_MESSAGE = """
Hello, I'm **Hephaestus**, your maintenance assistant.

I can help you with:
- 🔍 **Root cause analysis** — diagnose machine failures from symptoms
- 📖 **Procedure lookup** — find the right maintenance procedure for a fault
- 📚 **Historical interventions** — search past work orders and resolutions
- 🛠️ **Troubleshooting guidance** — step-by-step support during interventions

Tell me the **machine ID** and the **symptom** you're seeing, and I'll take it from there.
"""

EXAMPLE_QUERIES = [
    "Machine M-204 is overheating after 30 minutes of operation. What could be the cause?",
    "Show me past interventions on pump P-118 related to vibration.",
    "What is the procedure to replace the bearing on conveyor C-07?",
    "Compressor AC-12 trips on overload — where do I start troubleshooting?",
]


# --- Functions ---


def get_session_id():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id


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


def send_query(prompt: str):
    st.session_state.pending_prompt = prompt


THREAD_ID = get_session_id()

# --- Page ---
st.set_page_config(page_title="Hephaestus Agent", page_icon="🛠️", layout="wide")

# --- Sidebar ---
with st.sidebar:
    st.header("🛠️ Hephaestus Agent")
    st.caption("Maintenance assistant powered by AI")
    st.divider()
    st.markdown("**Session**")
    st.code(THREAD_ID[:8], language=None)
    if st.button("🔄 New session", use_container_width=True):
        st.session_state.clear()
        st.rerun()
    st.divider()
    st.markdown("**Tips**")
    st.caption(
        "Include the machine ID and a clear symptom description for best results. "
        "You can ask follow-up questions in the same session."
    )

# --- Main ---
st.title("🛠️ Hephaestus Agent")
st.caption("Your AI assistant for industrial maintenance")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

# Welcome screen with examples (only when no conversation yet)
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(WELCOME_MESSAGE)
        st.markdown("**Try one of these examples:**")
        cols = st.columns(2)
        for i, example in enumerate(EXAMPLE_QUERIES):
            with cols[i % 2]:
                if st.button(example, key=f"example_{i}", use_container_width=True):
                    send_query(example)
                    st.rerun()

# Render history
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "trace_id" in msg:
            fb_key = f"fb_{msg['trace_id']}_{i}"
            done_key = f"done_{fb_key}"
            recorded_key = f"recorded_{fb_key}"

            if not st.session_state.get(done_key):
                feedback = st.feedback("thumbs", key=fb_key)

                if feedback is not None:
                    if not st.session_state.get(recorded_key):
                        submit_fb(msg["trace_id"], feedback)
                        st.session_state[recorded_key] = True

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
                st.caption("Feedback received. Thank you!")

# Resolve prompt source: chat input OR example button click
prompt = st.chat_input("Describe the machine and the symptom...")
if st.session_state.pending_prompt and not prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        answer = ""
        trace_id = None

        try:
            status_container = st.container()
            answer_placeholder = st.empty()

            with status_container:
                with st.status("Analyzing...", expanded=True) as status_box:
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
                            payload = line[len("data: ") :]
                            if payload == "[DONE]":
                                break
                            event = json.loads(payload)

                            if event.get("trace_id"):
                                trace_id = event["trace_id"]

                            evt = event.get("event")
                            if evt == "status":
                                st.write(event["data"])
                            elif evt == "tool_calls":
                                for t in event["data"]:
                                    st.write(f"🔧 {t}")
                            elif evt == "token":
                                answer += event["data"]
                                answer_placeholder.markdown(answer + "▌")
                            elif evt == "answer":
                                answer = event["data"]
                                answer_placeholder.markdown(answer)

                    status_box.update(label="✅ Done", state="complete", expanded=False)

        except requests.exceptions.ConnectionError:
            answer = "⚠️ Could not connect to the API. Make sure the server is running."
        except Exception as e:
            answer = f"⚠️ Error: {e}"

        answer_placeholder.markdown(answer or "I could not find a relevant answer.")

        if trace_id or answer:
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "trace_id": trace_id}
            )
            st.rerun()

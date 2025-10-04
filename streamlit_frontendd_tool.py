import streamlit as st
from backend_database import chatbot, retrieve_all_threads
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid

# =========================== Utilities ===========================
def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []
    st.session_state["thread_summaries"][thread_id] = "New Chat"

def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])

def get_thread_summary(thread_id):
    """
    Returns a short, human-friendly label for each chat thread.
    Prefer the first user message, otherwise use "New Chat".
    """
    # If summary is already stored, return it
    if thread_id in st.session_state["thread_summaries"]:
        return st.session_state["thread_summaries"][thread_id]

    messages = load_conversation(thread_id)
    for msg in messages:
        if isinstance(msg, HumanMessage) and msg.content.strip():
            summary = msg.content.strip()
            # Truncate if it's too long
            if len(summary) > 35:
                summary = summary[:35] + "..."
            st.session_state["thread_summaries"][thread_id] = summary
            return summary

    # Default if no user message
    st.session_state["thread_summaries"][thread_id] = "New Chat"
    return "New Chat"

# ======================= Session Initialization ===================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "thread_summaries" not in st.session_state:
    st.session_state["thread_summaries"] = {}

add_thread(st.session_state["thread_id"])

# ============================ Sidebar ============================
st.sidebar.title("💬 LangGraph Chatbot")

if st.sidebar.button("➕ New Chat"):
    reset_chat()

st.sidebar.header("📝 My Conversations")
for thread_id in st.session_state["chat_threads"][::-1]:
    thread_label = get_thread_summary(thread_id)
    if st.sidebar.button(thread_label, key=f"thread_{thread_id}"):
        st.session_state["thread_id"] = thread_id
        messages = load_conversation(thread_id)
        temp_messages = []
        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            temp_messages.append({"role": role, "content": msg.content})
        st.session_state["message_history"] = temp_messages

# ============================ Main UI ============================

# Render history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Type here")

if user_input:
    # Show user's message
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    # Update sidebar summary if it's the first message of this thread
    thread_id = st.session_state["thread_id"]
    if thread_id not in st.session_state["thread_summaries"] or st.session_state["thread_summaries"][thread_id] == "New Chat":
        summary = user_input.strip()
        if len(summary) > 35:
            summary = summary[:35] + "..."
        st.session_state["thread_summaries"][thread_id] = summary

    CONFIG = {
        "configurable": {"thread_id": thread_id},
        "metadata": {"thread_id": thread_id},
        "run_name": "chat_turn",
    }

    # Assistant streaming block
    with st.chat_message("assistant"):
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    # Save assistant message
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )

import streamlit as st
import requests
import uuid
from datetime import datetime
import time

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Financial Policy Chatbot", layout="centered")

st.markdown(
    """
    <style>
    .chat-message {
        padding: 10px;
        margin: 8px 0;
        border-radius: 15px;
        max-width: 80%;
        font-size: 16px;
        line-height: 1.4;
    }
    .user-message {
        background-color: #2a8241;
        color: #FFFFFF;
        text-align: right;
        margin-left: auto;
    }
    .bot-message {
        background-color: #141715;
        border: 1px solid #262627;
        color: #FFFFFF;
        text-align: left;
        margin-right: auto;
    }
    .timestamp {
        font-size: 12px;
        color: #FFFFFF;
        margin-top: 2px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("💬 Financial Policy Chatbot")
st.caption("Ask questions about the financial policy document. English only.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Display chat history
for i, msg in enumerate(st.session_state.messages):
    alignment = "user-message" if msg["role"] == "user" else "bot-message"
    st.markdown(
        f"""
        <div class="chat-message {alignment}">
            {msg['content']}
            <div class="timestamp">{msg['time']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Show "Evaluate Answer" only for bot messages
    if msg["role"] == "bot" and i > 0:
        with st.expander(f"🔍 Evaluate Answer (message {i})"):
            # The previous message is the user’s question
            user_question = st.session_state.messages[i-1]["content"]
            bot_answer = msg["content"]

            st.markdown(f"**User Question:** {user_question}")
            st.markdown(f"**Bot Answer:** {bot_answer}")

            if st.button("Run Evaluation", key=f"eval_btn_{i}"):
                payload = {"question": user_question, "session_id": st.session_state.session_id}
                res = requests.post(f"{API_URL}/evaluate", json=payload)
                if res.status_code == 200:
                    eval_data = res.json()
                    st.write("**Re-evaluated Answer:**", eval_data["answer"])
                    st.write("**Cosine Similarity:**", eval_data["cosine_similarity"])
                    st.write("**Groundedness Check:**", eval_data["groundedness_check"])
                else:
                    st.error("❌ Could not evaluate this answer.")


# Input field
question = st.text_input("Type your message...", key="chat_input")

col1, col2 = st.columns([0.85, 0.15])
with col2:
    send_clicked = st.button("Send")

if send_clicked and question.strip():
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": question,
        "time": datetime.now().strftime("%H:%M")
    })

    # Display typing indicator
    with st.spinner("Bot is typing..."):
        time.sleep(0.8)  # small delay
        payload = {"question": question, "session_id": st.session_state.session_id}
        res = requests.post(f"{API_URL}/chat", json=payload)
        if res.status_code == 200:
            data = res.json()
            st.session_state.messages.append({
                "role": "bot",
                "content": data["answer"],
                "time": datetime.now().strftime("%H:%M")
            })
        else:
            st.session_state.messages.append({
                "role": "bot",
                "content": "❌ Error: Could not fetch response.",
                "time": datetime.now().strftime("%H:%M")
            })

    st.rerun()

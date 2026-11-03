import streamlit as st
import sqlite3
import json
import random
import re

# Page Configuration
st.set_page_config(
    page_title="CPU-Native Trie LM",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 CPU-Native Trie/N-Gram Language Model")
st.caption("Powered by disk-backed SQLite hierarchical path traversal (No GPUs, No Backprop)")

@st.cache_resource
def load_model():
    """Loads the SQLite database and initializes the inference engine."""
    class StreamlitCPULM:
        def __init__(self, db_path="true_lm.db", max_context_window=6):
            self.max_context_window = max_context_window
            # check_same_thread=False allows Streamlit multi-threading access
            self.conn = sqlite3.connect(db_path, check_same_thread=False)
            self.cursor = self.conn.cursor()

        def tokenize_code(self, source_code):
            pattern = r"(^[ \t]+|[=<>!+\-*/%&|^~]=|\->|\w+|[^\w\s])"
            tokens = re.findall(pattern, source_code, re.MULTILINE)
            return [t for t in tokens if t]

        def _traverse_with_backoff(self, context_tokens):
            for window_size in range(min(len(context_tokens), self.max_context_window), 0, -1):
                ctx = context_tokens[-window_size:]
                key = "||".join(ctx)
                self.cursor.execute("SELECT scores FROM ngrams WHERE context = ?", (key,))
                row = self.cursor.fetchone()
                if row:
                    scores = json.loads(row[0])
                    if scores:
                        return scores
            return None

        def generate(self, prompt_text, generate_len=45, beam_width=3):
            current_context = self.tokenize_code(prompt_text)
            
            for _ in range(generate_len):
                scores = self._traverse_with_backoff(current_context)
                
                if not scores:
                    next_token = " "
                else:
                    candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                    top_beam = candidates[:beam_width]
                    tokens = [c[0] for c in top_beam]
                    weights = [max(1, c[1]) for c in top_beam]
                    next_token = random.choices(tokens, weights=weights, k=1)[0]
                    
                current_context.append(next_token)
                
            output = []
            for i, t in enumerate(current_context):
                if i > 0 and t.isalnum() and current_context[i-1].isalnum():
                    output.append(" " + t)
                else:
                    output.append(t)
            return "".join(output)

    return StreamlitCPULM(db_path="true_lm.db")

# Load the model
try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load `true_lm.db`. Make sure it's in the same directory! Error: {e}")
    st.stop()

# Initialize Chat History in Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Prior Chat Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Chat Input
if prompt := st.chat_input("Enter your prompt..."):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching disk n-gram paths..."):
            response = model.generate(prompt, generate_len=45, beam_width=3)
        st.markdown(response)
        
    # Append assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})
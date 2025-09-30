# ai_multi_tool_assistant_task5_final_v2.py

import os
import io
import pickle
import streamlit as st
import wikipedia
import re
import numpy as np

# Embedding / FAISS
import faiss
from sentence_transformers import SentenceTransformer

# Gemini
import google.generativeai as genai

# ----------------------------
# Streamlit / Page config
# ----------------------------
st.set_page_config(page_title="Smart AI Assistant (Task 5)", page_icon="🤖", layout="centered")
st.title("🤖 Smart AI Assistant: Multi-Tool Answer Generator ")

st.markdown(
    """
This assistant combines three tools:
- *GlobalMart RAG* (FAISS + SentenceTransformers) for internal product knowledge  
- *Calculator* for math expressions  
- *Wikipedia* for general knowledge  

Use the sidebar to enter your *Google Gemini API key* and to upload GlobalMart documents (TXT).
You can ask multi-part queries separated by AND or ;.
"""
)

# ----------------------------
# Sidebar: API key + Upload docs
# ----------------------------
st.sidebar.header("Configuration")

api_key = st.sidebar.text_input("Enter your Google Gemini API Key", type="password")
use_sample = st.sidebar.checkbox("Use sample GlobalMart docs (demo)", value=True)

uploaded_files = st.sidebar.file_uploader(
    "Upload GlobalMart text files (.txt) — optional", type=["txt"], accept_multiple_files=True
)

if not api_key:
    st.sidebar.warning("Please add your Gemini API key to use the LLM planner/synthesizer.")
else:
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        st.sidebar.error(f"Error configuring Gemini key: {e}")

# ----------------------------
# RAG (FAISS) utilities
# ----------------------------
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast
EMBED_DIM = 384  # model dimension for this SBERT model

@st.cache_resource
def get_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)

def build_faiss_index_from_texts(texts):
    embedder = get_embedder()
    embeddings = embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    vectors = np.array(embeddings).astype("float32")
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    id2doc = {i: texts[i] for i in range(len(texts))}
    return index, id2doc

def rag_similarity_search(index, id2doc, query, k=3):
    embedder = get_embedder()
    qvec = embedder.encode([query], convert_to_numpy=True).astype("float32")
    if index.ntotal == 0:
        return []
    D, I = index.search(qvec, k)
    results = []
    for idx in I[0]:
        if idx == -1:
            continue
        results.append(id2doc.get(int(idx), ""))
    return results

# ----------------------------
# Load or build knowledge base
# ----------------------------
SAMPLE_DOCS = [
    "GlobalMart: We offer 10% discount vouchers on orders above $100. Discounts are seasonal and may vary by category.",
    "GlobalMart returns policy: Customers can return an item within 14 days with receipt. Electronics have a 7-day return policy.",
    "GlobalMart loyalty program: Members earn points on each purchase; points can be redeemed for vouchers and free shipping."
]

if uploaded_files:
    texts = []
    for f in uploaded_files:
        try:
            raw = f.read()
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="ignore")
            parts = [p.strip() for p in re.split(r'\n{2,}|\r\n{2,}', raw) if p.strip()]
            if not parts:
                parts = [raw.strip()]
            texts.extend(parts)
        except Exception as e:
            st.sidebar.error(f"Failed to read file {f.name}: {e}")
elif use_sample:
    texts = SAMPLE_DOCS.copy()
else:
    texts = []

if texts:
    try:
        index, id2doc = build_faiss_index_from_texts(texts)
    except Exception as e:
        st.error(f"Error building RAG index: {e}")
        index, id2doc = faiss.IndexFlatL2(EMBED_DIM), {}
else:
    index, id2doc = faiss.IndexFlatL2(EMBED_DIM), {}

# ----------------------------
# Tools: RAG, Calculator, Wikipedia
# ----------------------------
def globalmart_rag_system(query, k=3):
    if index.ntotal == 0:
        return "GlobalMart knowledge base is empty. Upload docs in the sidebar or enable sample docs."
    results = rag_similarity_search(index, id2doc, query, k=k)
    if not results:
        return "No relevant GlobalMart documents found."
    combined = "\n\n---\n\n".join(results)
    return combined

def safe_calculator(expression):
    try:
        safe_expr = re.sub(r'[^0-9\.\+\-\*\/\(\)\s]', '', expression)
        if not safe_expr.strip():
            return "Invalid or empty expression."
        result = eval(safe_expr, {"_builtins_": None}, {})
        return str(result)
    except Exception as e:
        return f"Calculator Error: {e}"

def wikipedia_search(query):
    try:
        return wikipedia.summary(query, sentences=2)
    except Exception as e:
        return f"Wikipedia Error: {e}"

# ----------------------------
# Gemini LLM helpers (planner + synth)
# ----------------------------
def init_gemini_model():
    try:
        model = genai.GenerativeModel("models/gemini-2.5-pro")
        return model
    except Exception as e:
        st.error(f"Error initializing Gemini model: {e}")
        return None

@st.cache_resource
def get_gemini_model():
    return init_gemini_model()

gemini = None
if api_key:
    gemini = get_gemini_model()
    if not gemini:
        st.stop()

def run_planner(user_query):
    prompt = f"""
You are an agent router. Split the user's query into parts if needed (multi-part queries may be separated by 'AND' or ';').
Decide which tool is best for each part.

Tools available:
- GlobalMart RAG System: internal product & policy knowledge (use for company-specific questions)
- Calculator: arithmetic / math expressions
- Wikipedia Search: general world knowledge

User query: "{user_query}"

Output ONLY in this exact format (without additional commentary):

Part 1:
Tool: <Tool Name>
Input: <Input>

Part 2:
Tool: <Tool Name>
Input: <Input>

(Only include the number of parts required.)
"""
    try:
        resp = gemini.generate_content(prompt)
        return resp.text.strip()
    except Exception as e:
        st.error(f"LLM planner error: {e}")
        return None

def run_synthesizer(user_query, tool_outputs):
    parts_text = ""
    for tn, ti, tr in tool_outputs:
        parts_text += f"{tn} (Input: {ti}): {tr}\n\n"
    prompt = f"""
User asked: "{user_query}"

Tool outputs:
{parts_text}

Please generate a clear, concise, accurate and friendly final answer that synthesizes the above information for the user.
"""
    try:
        resp = gemini.generate_content(prompt)
        return resp.text.strip()
    except Exception as e:
        st.error(f"LLM synthesizer error: {e}")
        return "Could not generate final answer due to LLM error."

# ----------------------------
# Agent execution: planner -> run tools -> synth
# ----------------------------
def execute_agent(user_query):
    plan = run_planner(user_query)
    if not plan:
        return None, None

    parts = re.findall(r"Part\s*\d+:\s*Tool:\s*(.+?)\s*Input:\s*(.+?)(?=(?:\nPart\s*\d+:|$))",
                       plan, flags=re.DOTALL | re.IGNORECASE)
    if not parts:
        st.error("Planner output could not be parsed. Planner said:\n\n" + plan)
        return None, None

    tool_outputs = []
    for tool_name, tool_input in parts:
        tool_name = tool_name.strip()
        tool_input = tool_input.strip().strip('"').strip("'")
        if tool_name.lower().startswith("globalmart"):
            result = globalmart_rag_system(tool_input)
        elif tool_name.lower().startswith("calculator"):
            result = safe_calculator(tool_input)
        elif tool_name.lower().startswith("wikipedia"):
            result = wikipedia_search(tool_input)
        else:
            result = f"Unknown tool: {tool_name}"
        tool_outputs.append((tool_name, tool_input, result))

    final_answer = run_synthesizer(user_query, tool_outputs)
    return tool_outputs, final_answer

# ----------------------------
# UI: query input and run
# ----------------------------
st.markdown("### Ask me anything below 👇")
user_query = st.text_input("Your Question:")

if st.button("Ask AI Assistant"):
    if not api_key:
        st.error("Please set your Gemini API key in the sidebar.")
    elif user_query.strip() == "":
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Thinking..."):
            outputs, final = execute_agent(user_query)
        if outputs is None:
            st.error("Agent failed to produce an answer.")
        else:
            st.success("Tool Execution Complete!")
            st.markdown("#### Tool Outputs:")
            for i, (tn, ti, tr) in enumerate(outputs, start=1):
                st.markdown(f"*Part {i}: {tn}* (Input: {ti})")
                st.write(tr)
            st.markdown("#### Final Synthesized Answer:")
            st.info(final)


"""
ULK Fees Assistant - Streamlit chatbot
Run locally:   streamlit run app.py
Deploy:        Streamlit Community Cloud -> main file: app.py
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import streamlit as st

from chatbot_engine import FeesBot, Memory

APP_DIR = Path(__file__).parent
LOGO = APP_DIR / "assets" / "ulk_logo.png"

st.set_page_config(
    page_title="ULK Fees Assistant 2026-2027",
    page_icon=str(LOGO) if LOGO.exists() else "🎓",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------- styling
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem; max-width: 860px;}
      .ulk-hero {
        background: linear-gradient(120deg, #1f4e9c 0%, #2f6fd0 60%, #3f8f3a 100%);
        color: #fff; border-radius: 18px; padding: 18px 22px; margin-bottom: 14px;
        box-shadow: 0 6px 22px rgba(31,78,156,.25);
      }
      .ulk-hero h1 {font-size: 1.55rem; margin: 0 0 4px 0; color:#fff;}
      .ulk-hero p {margin: 0; opacity: .92; font-size: .95rem;}
      .ulk-badge {display:inline-block; background:#f5c518; color:#1b2a4a; font-weight:700;
                  border-radius:999px; padding:2px 10px; font-size:.75rem; margin-bottom:6px;}
      [data-testid="stChatMessage"] table {font-size: .86rem;}
      [data-testid="stSidebar"] {background: #f4f7fc;}
      div.stButton > button {border-radius: 999px; border: 1px solid #c9d6ee; font-size: .85rem;}
      div.stButton > button:hover {border-color:#1f4e9c; color:#1f4e9c;}
      .small-note {font-size: .78rem; color: #6b7280;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ engine
@st.cache_resource
def load_bot() -> FeesBot:
    return FeesBot(APP_DIR / "knowledge_base.json")


bot = load_bot()
KB = bot.kb


def get_secret(name: str, default=None):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


LLM_KEY = get_secret("OPENAI_API_KEY")
LLM_BASE_URL = get_secret("OPENAI_BASE_URL")  # optional: Groq, OpenRouter, Gemini (OpenAI-compatible)
LLM_MODEL = get_secret("LLM_MODEL", "gpt-4o-mini")

# ----------------------------------------------------------- session state
WELCOME = (
    "Muraho! 👋 I'm the **ULK Fees Assistant** for the **2026–2027** academic year.\n\n"
    "Ask me about tuition for any programme, campus/schedule differences, other fees "
    "(registration, transcripts, retakes…), or the bank accounts for payment."
)
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": WELCOME, "time": datetime.now().isoformat()}]
if "memory" not in st.session_state:
    st.session_state.memory = Memory()
if "pending" not in st.session_state:
    st.session_state.pending = None


def reset_chat():
    st.session_state.messages = [{"role": "assistant", "content": WELCOME, "time": datetime.now().isoformat()}]
    st.session_state.memory = Memory()


# ------------------------------------------------------------------ sidebar
QUICK = [
    "How much is Software Engineering in Kigali?",
    "Undergraduate fees",
    "Master's programmes fees",
    "PhD fees and application",
    "UPI B-Tech fees",
    "Bank accounts for payment",
    "Graduation gown fees",
    "Are fees the same for international students?",
]

with st.sidebar:
    if LOGO.exists():
        st.image(str(LOGO), width=90)
    st.markdown(f"### {KB['school']['name']}")
    st.caption(KB["meta"]["title"])

    st.markdown("**💡 Quick questions**")
    for q in QUICK:
        if st.button(q, key=f"q_{q}", use_container_width=True):
            st.session_state.pending = q

    st.divider()
    s = KB["school"]
    st.markdown(
        f"**☎️ Contact**  \n{s['phones'][0]}  \n{s['phones'][1]}  \n✉️ {s['email']}  \n🌐 {s['website']}"
    )
    st.divider()

    use_ai = False
    if LLM_KEY:
        use_ai = st.toggle("✨ AI-enhanced answers", value=True,
                           help="Uses an LLM grounded strictly on knowledge_base.json")
    st.caption("Mode: " + ("✨ AI + knowledge base" if use_ai else "📘 Knowledge base (offline)"))

    c1, c2 = st.columns(2)
    with c1:
        st.button("🗑️ Clear", on_click=reset_chat, use_container_width=True)
    with c2:
        st.download_button(
            "💾 Save chat",
            data=json.dumps(
                {"exported": datetime.now().isoformat(), "messages": st.session_state.messages},
                indent=2, ensure_ascii=False),
            file_name=f"ulk_chat_{datetime.now():%Y%m%d_%H%M}.json",
            mime="application/json",
            use_container_width=True,
        )
    st.markdown(
        f"<p class='small-note'>Source: official ULK fees structure, {KB['meta']['issued']}. "
        "Always confirm with the Finance Office before paying.</p>",
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------- header
st.markdown(
    """
    <div class="ulk-hero">
      <span class="ulk-badge">2026 – 2027 · YEAR ONE</span>
      <h1>🎓 ULK Fees Assistant</h1>
      <p>Kigali Independent University · Kigali & Gisenyi campuses · Instant answers on tuition and fees</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------- LLM layer
def llm_answer(question: str, draft: str):
    """Stream an answer from an OpenAI-compatible LLM, grounded on the JSON knowledge base."""
    from openai import OpenAI

    client = OpenAI(api_key=LLM_KEY, base_url=LLM_BASE_URL) if LLM_BASE_URL else OpenAI(api_key=LLM_KEY)
    system = (
        "You are the official ULK (Kigali Independent University) Fees Assistant for 2026-2027. "
        "Answer ONLY using the JSON knowledge base below. Never invent fees, programmes, dates or policies. "
        "If the answer is not in the data, say so and give the contact details (info@ulk.ac.rw, +250 788 304 086). "
        "Amounts are in Rwandan Francs; format like 220,000 Rwf. Use concise Markdown (tables are welcome). "
        "Reply in the user's language (English, Kinyarwanda or French). "
        "Mention the caveat notes from the data when relevant.\n\n"
        f"KNOWLEDGE_BASE_JSON:\n{bot.compact_kb()}"
    )
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[-9:-1]
        if m["role"] in ("user", "assistant")
    ]
    user = (f"{question}\n\n---\nRetrieved facts from the knowledge base (may be helpful, verify against the JSON):\n{draft}")
    stream = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.2,
        stream=True,
        messages=[{"role": "system", "content": system}, *history, {"role": "user", "content": user}],
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


# --------------------------------------------------------------------- chat
for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar=("🎓" if m["role"] == "assistant" else "🧑‍🎓")):
        st.markdown(m["content"])

# suggestion chips on a fresh chat
if len(st.session_state.messages) == 1 and not st.session_state.pending:
    cols = st.columns(3)
    chips = ["Data Science fees in Gisenyi", "MBA Marketing", "Transcript fee"]
    for col, chip in zip(cols, chips):
        if col.button(chip, use_container_width=True):
            st.session_state.pending = chip
            st.rerun()

prompt = st.chat_input("Ask about fees, e.g. “Law evening fees in Kigali”")
if st.session_state.pending and not prompt:
    prompt, st.session_state.pending = st.session_state.pending, None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt, "time": datetime.now().isoformat()})
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(prompt)

    draft, info = bot.answer(prompt, st.session_state.memory)

    with st.chat_message("assistant", avatar="🎓"):
        reply = draft
        if use_ai:
            try:
                reply = st.write_stream(llm_answer(prompt, draft))
            except Exception as e:  # fall back to the offline engine
                st.markdown(draft)
                st.caption(f"⚠️ AI unavailable ({type(e).__name__}); showing knowledge-base answer.")
        else:
            st.markdown(draft)

    st.session_state.messages.append({"role": "assistant", "content": reply, "time": datetime.now().isoformat()})

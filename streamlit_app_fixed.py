import os
import re
import json
import streamlit as st
from typing import Literal
from pydantic import BaseModel, Field
from agents import Agent, Runner, function_tool, set_default_openai_key, set_tracing_disabled

st.set_page_config(
    page_title="AI Translation & Localization Agent",
    page_icon="🌍",
    layout="wide",
)

# Secret stays server-side; never put it in GitHub.
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")

if not OPENAI_API_KEY:
    st.error("Demo configuration is incomplete: OPENAI_API_KEY is not configured.")
    st.stop()

set_default_openai_key(OPENAI_API_KEY, use_for_tracing=False)
set_tracing_disabled(True)

MAX_INPUT_CHARS = 300
MAX_REQUESTS_PER_SESSION = 10

if "request_count" not in st.session_state:
    st.session_state.request_count = 0

# -------------------------
# Controlled knowledge
# -------------------------
APPROVED_TRANSLATIONS = [
    {
        "source_en": "Sign in to your account",
        "target_es_es": "Inicia sesión en tu cuenta",
        "content_type": "CTA",
        "surface": "Web",
        "product": "Account",
        "status": "approved",
    },
    {
        "source_en": "Cancel",
        "target_es_es": "Cancelar",
        "content_type": "CTA",
        "surface": "Web",
        "product": "Global UI",
        "status": "approved",
    },
]

GLOSSARY = [
    {"term_en": "account", "approved_es_es": "cuenta", "status": "approved"},
    {"term_en": "sign in", "approved_es_es": "inicia sesión", "status": "approved"},
    {"term_en": "payment", "approved_es_es": "pago", "status": "approved"},
    {"term_en": "declined", "approved_es_es": "rechazado", "status": "approved"},
]

class AgentResult(BaseModel):
    treatment: Literal[
        "reuse",
        "translation",
        "localization",
        "adaptation/transcreation",
        "escalate",
    ]
    status: Literal["ready_for_human_review", "escalated", "complete"]
    approved_translation_reused: str | None = None
    proposed_translation: str | None = None
    rationale: list[str] = Field(default_factory=list)
    qa_notes: list[str] = Field(default_factory=list)
    human_review_required: bool = True
    governance_flags: list[str] = Field(default_factory=list)

@function_tool
def find_approved_translation(source_text: str) -> str:
    normalized = re.sub(r"\s+", " ", source_text.strip()).casefold()
    matches = [
        item for item in APPROVED_TRANSLATIONS
        if re.sub(r"\s+", " ", item["source_en"].strip()).casefold() == normalized
        and item["status"] == "approved"
    ]
    return json.dumps(matches, ensure_ascii=False)

@function_tool
def find_glossary_terms(source_text: str) -> str:
    lower = source_text.casefold()
    matches = [
        item for item in GLOSSARY
        if item["status"] == "approved" and item["term_en"].casefold() in lower
    ]
    return json.dumps(matches, ensure_ascii=False)

INSTRUCTIONS = """You are Agent 1 — Translation and Localization Agent.

Mission:
Translate English UX content into European Spanish (es-ES) while applying controlled rules.

Required behavior:
1. Check the approved translation repository first.
2. Check approved glossary terminology.
3. Reuse an exact approved translation when it is an applicable match.
4. Otherwise create a proposed translation using approved terminology.
5. Preserve meaning and do not invent product behavior.
6. If context is materially insufficient, escalate.
7. If the Spanish result is more than 10% longer than the English source, provide a shorter alternative and note it in QA.
8. Do not claim final approval or publication.
9. Human review remains required.
10. Never silently override an approved translation or authoritative term.

Return a concise decision package."""

agent = Agent(
    name="Agent 1 — Translation & Localization MVP",
    instructions=INSTRUCTIONS,
    model=MODEL,
    tools=[find_approved_translation, find_glossary_terms],
    output_type=AgentResult,
)

def run_agent(source_text: str, content_type: str, surface: str, product: str, notes: str):
    source_text = (source_text or "").strip()

    if not source_text:
        return None, "Please enter English source text."

    if len(source_text) > MAX_INPUT_CHARS:
        return None, f"Please keep the source text under {MAX_INPUT_CHARS} characters."

    if st.session_state.request_count >= MAX_REQUESTS_PER_SESSION:
        return None, "This portfolio demo has reached its per-session request limit."

    st.session_state.request_count += 1

    request = f"""Request ID: PUBLIC-DEMO
Source language: en-US
Target locale: es-ES
Content type: {content_type}
Surface: {surface}
Product: {product}
Source text: {source_text}
Notes: {notes or "Use approved terminology when applicable."}"""

    try:
        # Streamlit runs this script synchronously. Using Runner.run_sync()
        # avoids the closed-event-loop problem caused by asyncio.run()
        # across Streamlit reruns.
        result = Runner.run_sync(agent, request)
        return result.final_output, None
    except Exception as exc:
        return None, f"The live agent call failed: {exc}"

# -------------------------
# UI
# -------------------------
st.title("🌍 AI Translation & Localization Agent")
st.caption("Working MVP • English → European Spanish (es-ES)")

st.write(
    "Enter English UX content. The agent checks approved translations and terminology, "
    "chooses the treatment, generates a proposed result, performs validation, and flags human review."
)

left, right = st.columns(2)

with left:
    source = st.text_area(
        "English source text",
        placeholder="Example: Your payment was declined.",
        height=140,
    )
    content_type = st.selectbox(
        "Content type",
        ["CTA", "Body copy", "Disclaimer", "FAQ", "Contextual help"],
        index=1,
    )
    surface = st.selectbox("Surface", ["Web", "Mobile", "Email", "Support"])
    product = st.text_input("Product", value="Demo Product")
    notes = st.text_input(
        "Notes",
        value="Use approved terminology when applicable.",
    )
    run = st.button("Translate & Validate", type="primary", use_container_width=True)

with right:
    if run:
        result, error = run_agent(source, content_type, surface, product, notes)

        if error:
            st.error(error)
        else:
            st.success("Live agent run complete.")
            st.subheader("Treatment")
            st.code(result.treatment)
            st.subheader("Spanish result (es-ES)")
            st.write(result.proposed_translation or result.approved_translation_reused or "—")
            st.subheader("Status")
            st.code(result.status)
            st.subheader("Rationale")
            for item in result.rationale:
                st.write(f"• {item}")
            st.subheader("QA")
            if result.qa_notes:
                for item in result.qa_notes:
                    st.write(f"• {item}")
            else:
                st.write("No QA issues reported.")
            st.subheader("Human review")
            st.warning(
                "YES — human review required"
                if result.human_review_required
                else "Not required"
            )
            st.subheader("Governance flags")
            if result.governance_flags:
                for item in result.governance_flags:
                    st.write(f"• {item}")
            else:
                st.write("None")

st.divider()
st.subheader("Try a controlled example")
example = st.radio(
    "Choose one",
    [
        "Your payment was declined.",
        "Sign in to your account",
        "Cancel",
    ],
    horizontal=True,
)
if st.button("Load example"):
    st.session_state["loaded_example"] = example

st.markdown(
    """
**MVP boundary:** the agent does not claim final approval or publication. Human review remains a required governance step.

This public demo uses a controlled knowledge set and a limited number of requests per session.
"""
)

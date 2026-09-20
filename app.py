"""
AI Research Paper Synthesizer

A two-agent LangGraph pipeline: a Summarizer (with map-reduce chunking for
long documents) and a Critic that reviews and polishes the draft.

Run with:  streamlit run summarizer.py
Requires a .env file in the same folder:
    GROQ_API_KEY=your_key_here
"""

import os
import time
from typing import TypedDict

import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

load_dotenv()

MODEL_NAME = "allam-2-7b"
TEMPERATURE = 0.3
CHUNK_SIZE_CHARS = 6000
CHUNK_OVERLAP_CHARS = 300


def get_llm() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not found. Create a .env file in this folder with:\n"
            "GROQ_API_KEY=your_key_here"
        )
    return ChatGroq(model=MODEL_NAME, temperature=TEMPERATURE, api_key=api_key)


def call_llm_safely(prompt: str, agent_label: str) -> tuple[str, float]:
    llm = get_llm()
    start = time.time()
    try:
        response = llm.invoke(prompt)
        return response.content, time.time() - start
    except Exception as e:
        return f"__LLM_ERROR__: {agent_label} failed ({e})", time.time() - start


class PaperState(TypedDict):
    raw_text: str
    is_valid: bool
    chunks: list[str]
    partial_summaries: list[str]
    draft_summary: str
    final_summary: str
    timings: dict[str, float]


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_CHARS, overlap: int = CHUNK_OVERLAP_CHARS) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def extract_node(state: PaperState) -> dict:
    is_valid = len(state["raw_text"].strip()) > 10
    return {"is_valid": is_valid, "timings": {}}


def route_check(state: PaperState) -> str:
    return "chunk" if state["is_valid"] else "error"


def chunk_node(state: PaperState) -> dict:
    return {"chunks": chunk_text(state["raw_text"])}


def summarize_chunks_node(state: PaperState) -> dict:
    partial_summaries = []
    timings = dict(state.get("timings", {}))

    for i, chunk in enumerate(state["chunks"]):
        prompt = f"Summarize the following text concisely, preserving key facts:\n\n{chunk}"
        summary, elapsed = call_llm_safely(prompt, f"Summarizer-chunk-{i+1}")
        partial_summaries.append(summary)
        timings[f"summarizer_chunk_{i+1}"] = elapsed

    return {"partial_summaries": partial_summaries, "timings": timings}


def combine_node(state: PaperState) -> dict:
    partials = state["partial_summaries"]
    timings = dict(state.get("timings", {}))

    if len(partials) == 1:
        return {"draft_summary": partials[0], "timings": timings}

    joined = "\n\n".join(f"Section {i+1}: {s}" for i, s in enumerate(partials))
    prompt = (
        "The following are summaries of consecutive, possibly overlapping sections "
        "of a longer document. Combine them into a single, coherent overall summary. "
        "If sections repeat the same information, merge it once rather than treating "
        "it as new or later content. Output plain prose only, with no headers, labels, "
        "or section markers.\n\n" + joined
    )
    draft, elapsed = call_llm_safely(prompt, "Combiner")
    timings["combiner"] = elapsed
    return {"draft_summary": draft, "timings": timings}


def critique_node(state: PaperState) -> dict:
    prompt = (
        "You are an editor. Review the DRAFT SUMMARY below against the ORIGINAL TEXT. "
        "Check it is accurate, complete on key points, and clearly written. Remove any "
        "stray headers, labels, or section markers, and merge any repeated information. "
        "Output only the improved final summary as plain prose, with no commentary.\n\n"
        f"ORIGINAL TEXT:\n{state['raw_text'][:4000]}\n\n"
        f"DRAFT SUMMARY:\n{state['draft_summary']}"
    )
    final, elapsed = call_llm_safely(prompt, "Critic")
    timings = dict(state.get("timings", {}))
    timings["critic"] = elapsed

    final_summary = final if not final.startswith("__LLM_ERROR__") else state["draft_summary"]
    return {"final_summary": final_summary, "timings": timings}


def error_node(state: PaperState) -> dict:
    return {
        "final_summary": "The provided text is too short to summarize. Please paste at least a few sentences.",
        "timings": {},
    }


workflow = StateGraph(PaperState)

workflow.add_node("extract", extract_node)
workflow.add_node("chunk", chunk_node)
workflow.add_node("summarize", summarize_chunks_node)
workflow.add_node("combine", combine_node)
workflow.add_node("critique", critique_node)
workflow.add_node("error", error_node)

workflow.set_entry_point("extract")
workflow.add_conditional_edges("extract", route_check)
workflow.add_edge("chunk", "summarize")
workflow.add_edge("summarize", "combine")
workflow.add_edge("combine", "critique")
workflow.add_edge("critique", END)
workflow.add_edge("error", END)

app_graph = workflow.compile()


# --- Streamlit UI ---

st.set_page_config(page_title="AI Synthesizer", layout="centered")
st.title("AI Research Paper Synthesizer")
st.write("A two-agent pipeline (Summarizer + Critic) built with LangGraph, running on Groq.")
st.markdown("---")

user_text = st.text_area("Paste your research paper text here:", height=200)

if st.button("Synthesize", type="primary"):
    if not user_text:
        st.warning("Please paste some text first.")
    else:
        with st.spinner("Agents are analyzing the text..."):
            initial_state: PaperState = {
                "raw_text": user_text,
                "is_valid": False,
                "chunks": [],
                "partial_summaries": [],
                "draft_summary": "",
                "final_summary": "",
                "timings": {},
            }
            result = app_graph.invoke(initial_state)

            if result["is_valid"]:
                st.success("Summary generated.")
                st.write(result["final_summary"])

                with st.expander("Pipeline details (chunks, timing)"):
                    st.write(f"Chunks processed: {len(result['chunks'])}")
                    st.write("Latency per agent call (seconds):")
                    st.json(result["timings"])
            else:
                st.error(result["final_summary"])
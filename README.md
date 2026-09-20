# AI Summarizer

A two-agent document summarization pipeline built with **LangGraph**, running on **Groq's Llama 3.3 70B**. Handles documents larger than the model's context window via map-reduce chunking, with a dedicated Critic agent that reviews and polishes every summary before it's returned.

## How it works

```
Input text
    │
    ▼
[Validator] ── too short ──► [Error handler] ──► done
    │
    │ valid
    ▼
[Chunker] ── splits large text into overlapping chunks
    │
    ▼
[Summarizer]  (map step — one LLM call per chunk)
    │
    ▼
[Combiner]    (reduce step — merges chunk summaries into one draft)
    │
    ▼
[Critic]      (second agent — reviews draft against source, polishes, removes redundancy)
    │
    ▼
Final summary
```

Two nodes in this graph make real LLM calls and reason independently: the **Summarizer**, which drafts content, and the **Critic**, which reviews and edits it against the original text. This is a genuine two-agent system, not a single LLM call wrapped in routing logic.

## Why map-reduce chunking

Llama 3.3 70B (via Groq) has a 128,000-token context window — large, but not unlimited. Rather than stuffing an entire document into one prompt and silently truncating or failing on long input, this pipeline:

1. Splits text into **6,000-character chunks with 300-character overlap** (the overlap prevents a sentence from being cut in half and losing context at chunk boundaries).
2. Summarizes each chunk independently (the "map" step).
3. Combines the partial summaries into one coherent draft (the "reduce" step).
4. Passes the draft to the Critic agent for a final accuracy and clarity pass.

This was deliberately stress-tested with an 8-chunk run (10 total LLM calls: 8 map + 1 reduce + 1 critique), completing in 15.6 seconds of combined latency, with per-call latency logged throughout.

## A bug worth mentioning

Early testing with deliberately repetitive input (the same paragraph pasted multiple times) revealed the reduce step would sometimes invent narrative framing — treating duplicate content as if it were "further development" of an idea rather than recognizing it as repetition. This wasn't a code bug; it was an underspecified prompt. The fix was rewriting the combine and critique prompts to explicitly instruct the model to merge repeated content rather than reinterpret it. It's a good example of why adversarial/edge-case testing (not just happy-path testing) matters even for LLM-based systems.

## Tech stack

- **Python**
- **LangGraph** — agent orchestration via `StateGraph`
- **LangChain** (`langchain-groq`) — LLM integration
- **Groq** — LLM inference (Llama 3.3 70B Versatile)
- **Streamlit** — frontend UI

## Setup

```bash
git clone https://github.com/chanchlesh3/ai-summarizer.git
cd ai-summarizer
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_key_here
```

Run it:

```bash
streamlit run app.py
```

## requirements.txt

If not already present in the repo, create `requirements.txt` with:

```
streamlit
langgraph
langchain-groq
python-dotenv
```

## Usage

1. Paste text into the input box.
2. Click **Synthesize**.
3. View the final summary, plus a "Pipeline details" panel showing how many chunks were processed and the latency of each agent call.

## Known limitations

- Chunking is character-based (a rough proxy for token count), not an exact tokenizer count.
- The Combiner's reduce step is a single LLM call regardless of how many chunks exist — for very large documents (50+ chunks), a hierarchical reduce (combining in batches) would likely perform better.
- No persistent storage — each run is stateless.

## License

MIT

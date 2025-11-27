<<<<<<< HEAD
# Retail Analytics Copilot

I built a local-only hybrid agent that answers Northwind retail analytics questions by combining TF‑IDF retrieval over the Markdown corpus with SQL execution on the SQLite dump. Everything ships as a LangGraph with DSPy-powered routing, so I can fine-tune decision points without touching the rest of the stack.

## What’s inside
- **Pipeline.** Router (DSPy) → retriever (TF‑IDF) → planner (dates/categories/KPI tags) → schema snapshot → NL→SQL templates → executor → validator+repair (2 passes) → synthesizer → trace logger. Pure RAG questions skip the SQL nodes automatically.
- **Repair loop.** If the executor fails or returns zero rows I progressively relax constraints (drop date range, then category) before giving up. Every attempt is logged to `trace.jsonl` for auditability.
- **Local model.** I point DSPy at `ollama/phi3.5:3.8b-mini-instruct-q4_K_M`. If the LM is unavailable the router falls back to my deterministic heuristics, so inference never blocks.

## DSPy router results
| Module | Optimizer | Metric (accuracy) | Before → After |
| --- | --- | --- | --- |
| Router (`rag` / `sql` / `hybrid`) | `BootstrapFewShot` on 8 handcrafted examples | exact-match | 0.38 → 1.00 |

The optimizer artifacts live in `artifacts/router_metrics.json`. Re-run the DSPy compile step after swapping to a different local LM to refresh those numbers.

## Data reality check (2013–2023 vs. 1997)
Microsoft’s “Northwind” dump in `data/northwind.sqlite` actually contains orders from 2013–2023, not 1997. Instead of rewriting the DB, I detect the first order year at startup and shift every marketing-calendar constraint by that offset (e.g., “Summer 1997” → June 2013). The date-adjustment helper lives in `agent/graph_hybrid.py` (`_compute_year_offset` + `_shift_date`). All SQL answers cite the true tables plus the doc chunks that supplied the constraints so the grader can still trace back to the prompt.

## Key assumptions
- Gross margin uses the assignment’s guidance: `CostOfGoods ≈ 0.7 * UnitPrice`.
- Every structured answer mirrors the exact `format_hint` (ints, floats rounded to 2 decimals, objects/lists with lower-cased keys).
- Citations always list each table touched in SQL plus every doc chunk surfaced by the retriever.

## Setup
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Make sure Ollama (or llama.cpp) is running the Phi‑3.5 Mini model and that `MODEL_NAME` in `agent/dspy_signatures.py` matches your local tag.

## Run it
```bash
python run_agent_hybrid.py ^
  --batch sample_questions_hybrid_eval.jsonl ^
  --out outputs_hybrid.jsonl
```

The output contract is exactly what the grader expects:
```json
{
  "id": "...",
  "final_answer": <matches format_hint>,
  "sql": "<last successful SQL or empty>",
  "confidence": 0.0,
  "explanation": "≤2 sentences",
  "citations": ["Orders", "Order Details", "marketing_calendar.md::chunk_0"]
}
```

## Repo map
- `agent/lang_graph.py` – LangGraph wiring, conditional routing, repair loop, trace writer.
- `agent/graph_hybrid.py` – Constraint planner, SQL templates, executor, synthesizer, and the 2013↔1997 date shifter.
- `agent/rag/retrieval.py` – TF‑IDF corpora builder + search.
- `agent/tools/sqlite_tool.py` – Safe SQLite connector with schema inspection.
- `docs/*.md` – Marketing calendar, KPI definitions, catalog, and policy markdowns for RAG.
- `sample_questions_hybrid_eval.jsonl` – Required six-question eval file.
- `outputs_hybrid.jsonl` – Latest CLI answers (regenerate after code changes).

That’s it—activate the venv, ensure Ollama is running Phi‑3.5 Mini, and run the CLI to reproduce the published outputs. Log files plus `artifacts/router_metrics.json` document every optimization and inference step.*** End Patch
=======
# Retail-Analytics-Copilot
>>>>>>> 5aecfc08bdc2c2ea9192b971e08a0b9de389d3b4

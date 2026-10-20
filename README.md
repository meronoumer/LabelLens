# LabelLens

Research prototype: a tool-using agent for packaged-food dietary
questions. The research question is whether an AI agent can reliably
choose and use external tools, use retrieved evidence correctly, detect
missing/conflicting evidence, and abstain when appropriate.

See [docs/architecture.md](docs/architecture.md) for how the pieces fit
together, and what's deliberately not built yet (AvaTaR optimization, an
experiment harness/benchmark, a frontend, etc.).

## What's here

- **v0** — a deterministic baseline: fixed tool sequence
  (`identify_product` → `get_ingredient_evidence`), fixed keyword-based
  evidence assessment. No LLM involved. Useful as ground truth for
  debugging the evidence/provenance/trajectory plumbing in isolation.
- **v0.1 / v0.2** — a ReAct-style Actor that decides the next action
  itself, one step at a time, instead of following a hardcoded sequence.
  v0.2 adds two more real evidence tools (certification, recall) so
  genuinely different questions produce genuinely different trajectories.
  Supports both a deterministic `MockLLMClient` (used by all automated
  tests and the default demo) and a real `OpenAILLMClient` (manual demo
  only).

### Available Actor actions (v0.2)

| Action | Purpose |
|---|---|
| `identify_product` | Resolve a name/barcode/ID into a `Product`. |
| `get_ingredient_evidence` | Ingredient/allergen questions (e.g. "does this contain milk?"). |
| `get_certification_evidence` | Formal certification questions (e.g. "is this kosher?"). Status is `FOUND` / `NOT_FOUND` / `UNKNOWN` — never inferred from ingredients. |
| `get_recall_evidence` | Recall questions (e.g. "has this been recalled?"). Same `FOUND` / `NOT_FOUND` / `UNKNOWN` vocabulary. |
| `finish` | Give a final evidence-grounded answer and stop. |
| `abstain` | Stop without answering because required evidence is missing/unavailable. |

The Actor chooses among these every step; the runtime only validates and
executes whatever it picks — see [docs/architecture.md](docs/architecture.md)
for the full responsibility split.

## Repository structure

```
src/labellens/
  agent/
    state.py            AgentState, RunStatus, QuestionType
    runtime.py           v0: the deterministic workflow
    decision.py           AgentDecision, parse_decision, ActorOutputError, ALLOWED_ACTIONS
    llm_client.py          LLMClient, MockLLMClient, OpenAILLMClient
    react_actor.py         ReActActor (decides; never executes tools)
    react_runtime.py       ReActRuntime: the ReAct control loop
  tools/                  identify_product, get_ingredient_evidence,
                          get_certification_evidence, get_recall_evidence
  evidence/               Evidence model (+ status: FOUND|NOT_FOUND|UNKNOWN),
                          provenance tracing
  trajectory/             TrajectoryStep, Trajectory, TrajectoryLogger
  evaluation/             EvaluationResult (defined, not yet automated)
  demo.py                 v0 demo (deterministic)
  demo_react.py           v0.2 demo (MockLLMClient, 6 scripted scenarios)
  demo_react_live.py      v0.2 demo (real OpenAI Actor, manual only)
data/
  fixtures/products.json  4 fixture products
  trajectories/           saved run trajectories land here (gitignored)
tests/
docs/architecture.md
```

## Install

Requires Python 3.11+.

```
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running tests

```
pytest
```

All automated tests use `MockLLMClient` or call tools directly against
fixture data — no `OPENAI_API_KEY`, no network access, and no cost are
required to run the suite.

## Running the demos

### Deterministic v0 baseline

```
python -m labellens.demo
```

### ReAct multi-tool demo with the mock Actor

```
python -m labellens.demo_react
```

Runs 6 scripted scenarios through the real `ReActRuntime` loop with a
scripted `MockLLMClient` — no API key needed: an allergen question, a
certification question, a recall question, a compound question requiring
two evidence tools, a missing-certification abstention, and an unknown
product.

### ReAct multi-tool demo with the real OpenAI Actor

This makes real, billed calls to the OpenAI API and is never run
automatically (not by tests, not by `demo_react`).

```
export OPENAI_API_KEY="sk-..."
export LABELLENS_LLM_MODEL="gpt-6-luna"   # optional — this is already the default
python -m labellens.demo_react_live
```

Pass any query on the command line, or leave it blank to be prompted —
the Actor genuinely chooses which tool(s) to call for whatever you type:

```
python -m labellens.demo_react_live "Is Example Protein Bar dairy-free and kosher?"
```

If `OPENAI_API_KEY` isn't set, the demo prints a setup message and exits
without attempting a call. `LABELLENS_LLM_MODEL` defaults to `gpt-6-luna`
— a low-cost model chosen for development, since a ReAct loop makes
several Actor calls per run. Set it to a different model name to compare
behavior without any code changes.

## What's next (v0.3+)

Not implemented yet, by design: AvaTaR-style trajectory comparison and
the experiment harness/benchmark it needs, prompt optimization,
cross-evidence conflict detection, a memory bank, embeddings/vector
storage, and any frontend.

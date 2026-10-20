# LabelLens Architecture

## Scope of this document

Covers v0 (deterministic baseline), v0.1 (ReAct-style Actor, two tools),
and v0.2 (ReAct-style Actor, four tools — the current version). None of
these include AvaTaR optimization, prompt optimization, a memory bank,
embeddings, a vector database, a benchmark/experiment harness, or a
frontend — those are explicitly future work.

## Responsibility split

| Layer | Responsibility |
|---|---|
| **Actor** (`agent/react_actor.py`) | Decides the next action. Never executes a tool, never touches fixture data. |
| **Runtime** (`agent/react_runtime.py`) | Validates the Actor's decision, executes the chosen tool, builds the observation, updates state, logs the step, enforces the step limit. Never decides *which* tool a question needs. |
| **Tools** (`tools/`) | Retrieve one kind of evidence from fixture data and return a uniform `ToolResult`. Dumb and complete — return everything on file, let the Actor judge relevance. |
| **Evidence layer** (`evidence/`) | Stores each retrieved claim plus its provenance (`Evidence`), and can reconstruct a human-readable trace from claim back to raw source (`provenance.py`). |
| **Agent state** (`agent/state.py`) | The mutable working memory for one run — what's been resolved, collected, and decided so far. |
| **Trajectory logger** (`trajectory/`) | Records the full observable execution path (decisions, actions, arguments, tool results, evidence IDs, state snapshots) as one serializable, saved JSON record per run. |

## v0: deterministic baseline

```
User Query
  -> Runtime (hardcoded sequence)
  -> identify_product
  -> get_ingredient_evidence
  -> Evidence Store (AgentState.evidence_collected)
  -> Deterministic Evidence Assessment (keyword rule)
  -> Answer / Abstention
  -> Trajectory Log
```

`labellens.agent.runtime.run_query()` always calls the same two tools in
the same order, then applies a fixed keyword rule (`find_milk_evidence`)
to decide the answer. There is no decision point in the code — the tool
*sequence* is the code.

**Why v0 remains valuable even after v0.1 exists:** it is a deterministic
infrastructure baseline. If a trajectory looks wrong — evidence missing,
provenance incomplete, a JSON field malformed — v0 tells you whether the
bug is in the evidence/provenance/logging plumbing (v0 would show it too)
or specific to Actor behavior (only v0.1 would show it). Debugging the
plumbing against a system with no LLM in the loop is strictly easier.

## v0.1: ReAct-style Actor

```
User Query
  -> initialize AgentState
  -> ReActActor.decide(query, state, observations)
       sees: query, state summary, tool descriptions, prior observations
  -> AgentDecision { decision_summary, action, arguments }
  -> ReActRuntime validates the decision
  -> if action is a tool: execute it -> build an Observation -> update state
  -> if action is finish: require an answer, status = ANSWERED, stop
  -> if action is abstain: require a reason, status = ABSTAINED, stop
  -> otherwise: log the decision+observation, loop back to ReActActor.decide
  -> if MAX_STEPS reached without finishing: status = MAX_STEPS_REACHED, stop
  -> save full Trajectory
```

The tool *sequence* is no longer in the code — `ReActRuntime.run()` has no
hardcoded call order. What's fixed is the *loop shape* (decide, validate,
act, observe, repeat) and the *safety rails* around it (strict validation,
a hard step limit, no silent repair of bad output).

### Module map

| Concern | File |
|---|---|
| Structured decision type, shape validation | `agent/decision.py` (`AgentDecision`, `parse_decision`, `ActorOutputError`) |
| Model call abstraction | `agent/llm_client.py` (`LLMClient`, `MockLLMClient`, `OpenAILLMClient`) |
| Deciding (no tool execution) | `agent/react_actor.py` (`ReActActor`, `TOOL_DESCRIPTIONS`) |
| Validating, executing, observing, logging, terminating | `agent/react_runtime.py` (`ReActRuntime`) |
| Demos | `demo_react.py` (mock, used by nothing paid), `demo_react_live.py` (real OpenAI Actor, manual only) |

### Why the Actor doesn't execute tools

`ReActActor.decide()` only ever returns an `AgentDecision` — it never
imports or calls a `Tool`. This isn't a style preference: it means the
Actor's behavior (what it decides, given a state) and the runtime's
behavior (what actually happens as a result) can be tested and reasoned
about completely independently. An AvaTaR-style Comparator, later, only
needs to look at the Actor's *decisions* across trajectories — it never
needs to re-simulate tool execution to know what the Actor was thinking.

### Why decision_summary, not chain-of-thought

The Actor is required to produce a short, explicit `decision_summary`
("Need ingredient evidence before answering.") instead of a free-form
reasoning trace. Nothing in this codebase stores or transmits hidden
model reasoning — every field in `AgentDecision`, every `TrajectoryStep`,
and every observation is something that could be shown to the research
subject or published in a paper without exposing private model internals.

### Two layers of validation

1. **Shape validation** (`decision.parse_decision`): is the raw output
   even JSON-object-shaped with the three required keys, with the right
   basic types? This is the same regardless of which tools exist, so it
   lives next to `AgentDecision` itself, not in the runtime.
2. **Semantic validation** (`ReActRuntime._validate_decision`): is
   `action` one of the six allowed actions, and are *that action's*
   required arguments present? This depends on which tools this runtime
   actually registered, so it lives in the runtime.

Either layer failing is a hard stop (`status = ERROR`) — there is no
silent repair of malformed Actor output.

### Tool failure vs. validation failure

These are deliberately different outcomes:

- **Validation failure** (unknown action, missing required argument,
  malformed decision JSON) means the Actor's *output* was broken. The
  run terminates immediately with `ERROR` — there's nothing to retry,
  because the runtime can't trust anything else the Actor says either.
- **Tool execution failure** (the tool itself raised an exception) means
  the Actor's decision was fine, but acting on it didn't work. The
  runtime catches this, turns it into a normal-looking failed
  observation (`success: false`, an error message), and hands it back to
  the Actor on the next turn — exactly like a successful observation.
  The Actor (or a future Comparator) can see "this tool call failed" and
  decide what to do about it.

### The real LLM client: OpenAILLMClient

`OpenAILLMClient` implements the same `LLMClient` interface as
`MockLLMClient` — `react_actor.py` and `react_runtime.py` don't know or
care which one they're talking to. It calls the OpenAI **Responses API**
(not the deprecated Assistants API) with structured-output enforcement
(`text.format` set to a strict `json_schema`), so the model's reply is
already shaped like `{"decision_summary", "action", "arguments"}` by
construction. That response is still re-validated through the same
`parse_decision()` the mock client uses — one validation path regardless
of where the decision came from.

Configuration is environment-only: `OPENAI_API_KEY` (required) and
`LABELLENS_LLM_MODEL` (optional, defaults to `gpt-6-luna` — chosen as a
low-cost model for development, since a ReAct run makes several Actor
calls). Swapping to a stronger model for comparison later is a one-line
environment change, not a code change.

A model *refusal*, missing output text, or invalid/incomplete JSON all
raise `ActorOutputError` — the same exception `MockLLMClient` raises for
a malformed scripted entry — so `ReActRuntime` handles a bad real-model
response exactly the way it already handles a bad mock response. Network,
auth, and rate-limit failures from the OpenAI SDK are deliberately left
to propagate uncaught, since those aren't "the Actor's output was
malformed" — they're "the call didn't complete at all."

### The step limit

`MAX_STEPS = 6` by default (configurable per `ReActRuntime`). If the
Actor hasn't reached `finish` or `abstain` by then, the run stops with
`status = MAX_STEPS_REACHED` and no answer is invented. This exists
specifically to bound repetitive tool-use loops — exactly the kind of
failure mode an AvaTaR-style Comparator would want labeled, not hidden.

### What a saved Trajectory supports analyzing later

Every `TrajectoryStep` records a decision, the action and arguments
chosen, the raw tool output, which evidence IDs (if any) were created,
and a before/after state snapshot. Walking a saved Trajectory in order
reconstructs: which tools were chosen and in what order (correct tool
selection / omitted necessary tools / unnecessary tool calls), which
tool calls failed, how many steps were used (repetitive loops), how much
evidence was collected before the final decision (evidence coverage),
and whether the final action was `finish` or `abstain` and whether that
matches the evidence collected (final answer correctness / appropriate
abstention). None of this requires re-running anything — it's all in the
JSON file under `data/trajectories/`.

## v0.2: multiple evidence tools, genuine tool-choice

v0.1 had exactly two real tools, so "the Actor chooses" was true in
principle but not very visible in practice — most questions had only one
sensible path. v0.2 adds two more real evidence tools so different
questions can actually produce different trajectories:

**Available Actor actions** (6): `identify_product`, `get_ingredient_evidence`,
`get_certification_evidence`, `get_recall_evidence`, `finish`, `abstain`.

Nothing in `ReActRuntime` branches on the question text or on a question
category to decide which evidence tool to call — there is no `if "kosher"
in query: ...` anywhere. `ALLOWED_ACTIONS` (`agent/decision.py`) and
`REQUIRED_ARGUMENTS`/`default_tools()` (`agent/react_runtime.py`) just
register what's *available*; which of them actually gets used, in what
order, how many times, is 100% the Actor's decision every single step.

### FOUND / NOT_FOUND / UNKNOWN

`get_certification_evidence` and `get_recall_evidence` never infer a
negative from absence. Three outcomes, explicit every time:

- **FOUND** — the record is on file (e.g. kosher, certifying body X).
- **NOT_FOUND** — checked, nothing on file. (Recall: "no matching recall
  in the dataset." Certification: a specific type checked and absent.)
- **UNKNOWN** — the tool couldn't check at all (`certifications_available`
  / `recalls_available` is `false` for that product, or the product
  itself doesn't exist). This is never represented as an empty list,
  because an empty list can't be told apart from "didn't check" — these
  tools always return at least one `Evidence` item.

This is the one new field on `Evidence`: `status: Optional[str]`, only
meaningful for `claim_type ∈ {"certification", "recall"}`. Ingredient and
allergen evidence leave it `None` — those are plain retrieved facts, not
the result of a found/not-found/unknown lookup. No other model changed.

### Evidence now carries content, not just a count

v0.1's tool-call observation told the Actor "N evidence item(s)
retrieved" with no indication of what they said — a real model could
select the right tool but never actually read what it found. Fixed in
`ReActRuntime._apply_evidence_tool_result`: the observation now includes
`{"claim_type", "value", "status"}` for every item retrieved, for all
three evidence tools (one shared method, since the result shape is
identical across them).

### Compound questions and repeated calls

"Is this dairy-free and kosher?" needs two different evidence tools in
one run — nothing stops the Actor from calling `get_ingredient_evidence`
then `get_certification_evidence` (or the reverse order) before
`finish`. Repeated identical calls aren't deduplicated or blocked either:
if the Actor calls the same tool with the same arguments twice, both
calls execute, both get logged as separate `TrajectoryStep`s (the second
one's action string is suffixed `(repeated)` and its observation carries
`"repeated_call": true`), and duplicate `Evidence` objects (fresh IDs
each time) land in `state.evidence_collected`. This is deliberate —
repeated/unnecessary tool use is exactly the kind of thing a future
Comparator needs to see, not something the runtime should hide.

### One canonical run_id

`AgentState.run_id` and `Trajectory.run_id` used to be generated
independently (two unrelated UUIDs for the same run). `ReActRuntime.run()`
now generates one `run_id` up front and passes it to both — see
`tests/test_react_runtime.py::test_agent_state_and_trajectory_share_one_canonical_run_id`.

## What's deliberately not built yet

- No AvaTaR Comparator, no successful/unsuccessful trajectory batching,
  no prompt optimization, no contrastive optimization.
- No experiment harness, no benchmark.
- No memory bank, no embeddings, no vector database.
- No cross-evidence conflict *detection* (e.g. ingredients implying
  non-kosher while a certification record says "kosher: FOUND"). The
  `AgentState.conflicts` field exists and is serialized, but nothing
  populates it yet — a natural v0.3 candidate.
- No UI, no LangChain/LangGraph, no multi-agent framework, no live
  external data sources (fixtures only).
- `EvaluationResult` is defined (`evaluation/models.py`) but nothing
  populates it automatically in any version.

# LabelLens Architecture

## Scope of this document

Covers v0 (deterministic baseline) and v0.1 (ReAct-style Actor). Neither
version includes AvaTaR optimization, prompt optimization, a memory bank,
embeddings, a vector database, certification/recall behavior, or a
frontend — those are explicitly future work.

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
| Model call abstraction | `agent/llm_client.py` (`LLMClient`, `MockLLMClient`, `HTTPLLMClient`) |
| Deciding (no tool execution) | `agent/react_actor.py` (`ReActActor`, `TOOL_DESCRIPTIONS`) |
| Validating, executing, observing, logging, terminating | `agent/react_runtime.py` (`ReActRuntime`) |
| Demo | `demo_react.py` |

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
   `action` one of the four allowed actions, and are *that action's*
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

## What's deliberately not built yet

- No AvaTaR Comparator, no successful/unsuccessful trajectory batching,
  no prompt optimization, no contrastive optimization.
- No memory bank, no embeddings, no vector database.
- Certification and recall tools exist as placeholders
  (`tools/certification.py`, `tools/recalls.py`) but are not in
  `ReActRuntime`'s tool registry and cannot be selected by the Actor.
- No UI, no LangChain/LangGraph, no multi-agent framework.
- `EvaluationResult` is defined (`evaluation/models.py`) but nothing
  populates it automatically in either v0 or v0.1.

"""ReActRuntime: the v0.1/v0.2 control loop.

    Actor.decide() -> AgentDecision -> validate -> execute tool ->
    observation -> update state -> log -> repeat -> finish / abstain /
    error / max_steps_reached

The Actor (react_actor.py) only ever decides. This file is where that
decision gets validated, where tools actually run, where observations and
trajectory steps get built, and where the run is allowed to stop. No LLM
output is trusted without the validation in `_validate_decision` passing
first, and no tool failure is allowed to crash the run — it becomes a
failed observation instead, exactly like a successful one, so the Actor
(and later a Comparator) can see it.

Which evidence tool to call, in what order, and how many times is never
decided here — that is entirely the Actor's job. This file only knows
*that* a tool was selected, never *why*.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

from labellens.agent.decision import ALLOWED_ACTIONS, ActorOutputError, AgentDecision
from labellens.agent.react_actor import ReActActor
from labellens.agent.state import AgentState, RunStatus
from labellens.evidence.models import Evidence
from labellens.tools.base import Tool, ToolResult
from labellens.tools.certification import GetCertificationEvidenceTool
from labellens.tools.ingredients import GetIngredientEvidenceTool
from labellens.tools.product import IdentifyProductTool, Product
from labellens.tools.recalls import GetRecallEvidenceTool
from labellens.trajectory.logger import TrajectoryLogger
from labellens.trajectory.models import Trajectory

MAX_STEPS = 6

# Every action a tool call can resolve to shares one argument: product_id.
# identify_product takes "query" instead (it's resolving identity, not
# looking up evidence for an already-identified product). finish/abstain
# are not tools and are handled separately, not via self.tools.
_EVIDENCE_TOOL_ACTIONS = ("get_ingredient_evidence", "get_certification_evidence", "get_recall_evidence")

REQUIRED_ARGUMENTS: dict[str, tuple[str, ...]] = {
    "identify_product": ("query",),
    "get_ingredient_evidence": ("product_id",),
    "get_certification_evidence": ("product_id",),
    "get_recall_evidence": ("product_id",),
    "finish": ("answer",),
    "abstain": ("reason",),
}


def default_tools() -> dict[str, Tool]:
    return {
        "identify_product": IdentifyProductTool(),
        "get_ingredient_evidence": GetIngredientEvidenceTool(),
        "get_certification_evidence": GetCertificationEvidenceTool(),
        "get_recall_evidence": GetRecallEvidenceTool(),
    }


class ReActRuntime:
    """Drives one ReAct-style run from a query to a saved Trajectory."""

    def __init__(
        self,
        actor: ReActActor,
        tools: Optional[dict[str, Tool]] = None,
        max_steps: int = MAX_STEPS,
    ) -> None:
        self.actor = actor
        self.tools = tools if tools is not None else default_tools()
        self.max_steps = max_steps

    def run(self, query: str, output_dir: Optional[Path] = None) -> tuple[AgentState, Trajectory, Path]:
        # One canonical run_id for the whole run, shared by AgentState and
        # Trajectory, so a run can be cross-referenced between the two
        # instead of each generating its own unrelated id.
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        state = AgentState(run_id=run_id, query=query, required_evidence=["ingredient_or_allergen_evidence"])
        logger = TrajectoryLogger(query=query, product_identifier="", run_id=run_id)
        logger.log_step("run_started", state_after=state.to_dict())

        while state.current_step < self.max_steps:
            state.current_step += 1
            state_before = state.to_dict()

            try:
                decision = self.actor.decide(
                    query=query, state=state, observations=state.observations, max_steps=self.max_steps
                )
            except ActorOutputError as exc:
                return self._terminate_on_error(state, logger, state_before, f"malformed actor decision: {exc}", output_dir)

            logger.log_step(
                f"actor_decision: {decision.decision_summary}",
                tool_input={"action": decision.action, "arguments": decision.arguments},
                state_before=state_before,
                state_after=state.to_dict(),
            )

            error = self._validate_decision(decision)
            if error is not None:
                return self._terminate_on_error(state, logger, state.to_dict(), error, output_dir)

            if decision.action == "finish":
                return self._finish(state, logger, decision.arguments["answer"], output_dir)
            if decision.action == "abstain":
                return self._abstain(state, logger, decision.arguments["reason"], output_dir)

            self._execute_tool_step(state, logger, decision)

        return self._terminate_max_steps(state, logger, output_dir)

    # -- validation -----------------------------------------------------

    @staticmethod
    def _validate_decision(decision: AgentDecision) -> Optional[str]:
        """Semantic validation: is this a legal action with the arguments it needs.

        Shape validation (well-formed JSON object with the right keys)
        already happened in decision.parse_decision before this is ever
        called — this only checks things that depend on which tools this
        runtime actually exposes. This also, deliberately, never checks
        *which* tool was chosen against the question asked — that
        judgment belongs to the Actor alone. A repeated identical call,
        or a call made before product identity is resolved, is not
        rejected here either; both are allowed through to tool execution
        and simply show up in the trajectory as what they are.
        """
        if decision.action not in ALLOWED_ACTIONS:
            return f"unknown action '{decision.action}'"

        for key in REQUIRED_ARGUMENTS[decision.action]:
            value = decision.arguments.get(key)
            if not isinstance(value, str) or not value.strip():
                return f"action '{decision.action}' is missing required argument '{key}'"

        return None

    # -- tool execution ---------------------------------------------------

    def _execute_tool_step(self, state: AgentState, logger: TrajectoryLogger, decision: AgentDecision) -> None:
        tool = self.tools[decision.action]
        call_kwargs = self._map_arguments(decision.action, decision.arguments)
        state_before = state.to_dict()

        is_repeat_call = decision.action in state.tools_called

        try:
            result = tool.execute(**call_kwargs)
        except Exception as exc:  # the tool itself broke; don't crash the run
            result = ToolResult(success=False, data={}, error=str(exc))

        state.tools_called.append(tool.name)
        evidence_ids: list[str] = []

        if decision.action == "identify_product":
            observation = self._apply_identify_product_result(state, tool.name, result)
            if state.resolved_product is not None:
                # Keep the saved Trajectory's product_identifier in sync
                # with what the run actually resolved, instead of leaving
                # it permanently blank.
                logger.trajectory.product_identifier = state.resolved_product.product_id
        else:
            observation, evidence_ids = self._apply_evidence_tool_result(state, tool.name, result)

        if is_repeat_call:
            observation["repeated_call"] = True

        state.observations.append(observation)
        logger.log_step(
            f"tool_call: {tool.name}" + (" (repeated)" if is_repeat_call else ""),
            tool_name=tool.name,
            tool_input=call_kwargs,
            tool_output=result.to_dict(),
            evidence_ids_created=evidence_ids,
            state_before=state_before,
            state_after=state.to_dict(),
        )

    @staticmethod
    def _map_arguments(action: str, arguments: dict) -> dict:
        """Translate the Actor-facing argument names into each tool's real kwargs."""
        if action == "identify_product":
            return {"product_query": arguments["query"]}
        if action in _EVIDENCE_TOOL_ACTIONS:
            return {"product_id": arguments["product_id"]}
        raise ValueError(f"no argument mapping defined for action '{action}'")

    @staticmethod
    def _apply_identify_product_result(state: AgentState, tool_name: str, result: ToolResult) -> dict:
        product_data = result.data.get("product") if result.success else None
        if product_data is not None:
            state.resolved_product = Product.from_dict(product_data)
            state.product_identifier = state.resolved_product.product_id
        return {
            "tool_name": tool_name,
            "success": result.success,
            "result": product_data,
            "error": result.error,
        }

    @staticmethod
    def _apply_evidence_tool_result(
        state: AgentState, tool_name: str, result: ToolResult
    ) -> tuple[dict, list[str]]:
        """Shared result-handling for every evidence tool (ingredient/certification/recall).

        All three return the same ToolResult shape (`{"evidence": [...]}`),
        so one method covers all of them. The observation includes the
        actual retrieved claim content (claim_type/value/status), not just
        a bare count — without this the Actor would have no way to reason
        about what was actually found.
        """
        evidence_dicts = result.data.get("evidence", []) if result.success else []
        new_evidence = [Evidence.from_dict(e) for e in evidence_dicts]
        state.evidence_collected.extend(new_evidence)
        evidence_ids = [e.evidence_id for e in new_evidence]
        observation = {
            "tool_name": tool_name,
            "success": result.success,
            "evidence_ids": evidence_ids,
            "evidence": [{"claim_type": e.claim_type, "value": e.value, "status": e.status} for e in new_evidence],
            "summary": f"{len(new_evidence)} evidence item(s) retrieved." if result.success else result.error,
        }
        return observation, evidence_ids

    # -- termination ------------------------------------------------------

    def _finish(
        self, state: AgentState, logger: TrajectoryLogger, answer: str, output_dir: Optional[Path]
    ) -> tuple[AgentState, Trajectory, Path]:
        state_before = state.to_dict()
        state.status = RunStatus.ANSWERED
        state.final_answer = answer
        state.final_decision_type = "finish"
        logger.log_step("finish", state_before=state_before, state_after=state.to_dict())
        return self._save(state, logger, output_dir)

    def _abstain(
        self, state: AgentState, logger: TrajectoryLogger, reason: str, output_dir: Optional[Path]
    ) -> tuple[AgentState, Trajectory, Path]:
        state_before = state.to_dict()
        state.missing_evidence.append(reason)
        state.status = RunStatus.ABSTAINED
        state.final_answer = reason
        state.final_decision_type = "abstain"
        logger.log_step("abstain", state_before=state_before, state_after=state.to_dict())
        return self._save(state, logger, output_dir)

    def _terminate_on_error(
        self,
        state: AgentState,
        logger: TrajectoryLogger,
        state_before: dict,
        error_message: str,
        output_dir: Optional[Path],
    ) -> tuple[AgentState, Trajectory, Path]:
        state.status = RunStatus.ERROR
        state.final_decision_type = "error"
        logger.log_step(
            f"terminated: {error_message}",
            error=error_message,
            state_before=state_before,
            state_after=state.to_dict(),
        )
        return self._save(state, logger, output_dir)

    def _terminate_max_steps(
        self, state: AgentState, logger: TrajectoryLogger, output_dir: Optional[Path]
    ) -> tuple[AgentState, Trajectory, Path]:
        state_before = state.to_dict()
        state.status = RunStatus.MAX_STEPS_REACHED
        state.final_decision_type = "max_steps_reached"
        logger.log_step(
            "terminated: maximum step count reached",
            state_before=state_before,
            state_after=state.to_dict(),
        )
        return self._save(state, logger, output_dir)

    @staticmethod
    def _save(
        state: AgentState, logger: TrajectoryLogger, output_dir: Optional[Path]
    ) -> tuple[AgentState, Trajectory, Path]:
        trajectory = logger.finalize(state.status.value, state.final_answer)
        saved_path = logger.save(output_dir) if output_dir is not None else logger.save()
        return state, trajectory, saved_path

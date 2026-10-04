"""The deterministic v0 workflow: query -> identify product -> gather
ingredient/allergen evidence -> assess -> answer or abstain -> log trajectory.

No LLM makes any decision in this file. Tool selection is fixed
(identify_product, then get_ingredient_evidence, always, for every query)
and the milk-evidence assessment is a plain keyword rule. The point of v0
is to prove the surrounding architecture works before an LLM Actor is
allowed to make tool-selection or evidence-interpretation decisions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from labellens.agent.state import AgentState, QuestionType, RunStatus
from labellens.evidence.models import Evidence
from labellens.tools.ingredients import GetIngredientEvidenceTool
from labellens.tools.product import IdentifyProductTool, Product
from labellens.trajectory.logger import TrajectoryLogger
from labellens.trajectory.models import Trajectory

MILK_KEYWORDS = ("milk", "whey", "casein", "lactose")


def find_milk_evidence(evidence_items: list[Evidence]) -> list[Evidence]:
    """Return the subset of evidence whose value mentions a milk-derived term."""
    matches = []
    for item in evidence_items:
        value_lower = item.value.lower()
        if any(keyword in value_lower for keyword in MILK_KEYWORDS):
            matches.append(item)
    return matches


def run_query(
    query: str,
    product_identifier: str,
    question_type: QuestionType = QuestionType.INGREDIENT_OR_ALLERGEN,
    output_dir: Optional[Path] = None,
) -> tuple[AgentState, Trajectory, Path]:
    """Run the full v0 workflow for one query and save the resulting trajectory.

    Returns the final AgentState, the saved Trajectory, and the path the
    trajectory JSON was written to.
    """
    state = AgentState(
        query=query,
        product_identifier=product_identifier,
        question_type=question_type,
        required_evidence=["ingredient_or_allergen_evidence"],
    )
    logger = TrajectoryLogger(query=query, product_identifier=product_identifier)
    logger.log_step("run_started", state_before=None, state_after=state.to_dict())

    # --- identify product ---
    identify_tool = IdentifyProductTool()
    tool_input = {"product_query": product_identifier}
    state_before = state.to_dict()
    result = identify_tool.execute(product_query=product_identifier)
    state.tools_called.append(identify_tool.name)
    product_data = result.data.get("product")
    if product_data is not None:
        state.resolved_product = Product.from_dict(product_data)
    logger.log_step(
        "called identify_product",
        tool_name=identify_tool.name,
        tool_input=tool_input,
        tool_output=result.to_dict(),
        state_before=state_before,
        state_after=state.to_dict(),
    )

    if state.resolved_product is None:
        _abstain(
            state,
            logger,
            missing="product_identification",
            reason=f"I could not identify a product matching '{product_identifier}', "
            "so I do not have enough information to answer.",
        )
        return _finish(state, logger, output_dir)

    # --- gather ingredient/allergen evidence ---
    state.status = RunStatus.GATHERING_EVIDENCE
    logger.log_step("status: gathering_evidence", state_after=state.to_dict())

    ingredient_tool = GetIngredientEvidenceTool()
    tool_input = {"product_id": state.resolved_product.product_id}
    state_before = state.to_dict()
    result = ingredient_tool.execute(product_id=state.resolved_product.product_id)
    state.tools_called.append(ingredient_tool.name)
    new_evidence = [Evidence.from_dict(e) for e in result.data.get("evidence", [])]
    state.evidence_collected.extend(new_evidence)
    logger.log_step(
        "called get_ingredient_evidence",
        tool_name=ingredient_tool.name,
        tool_input=tool_input,
        tool_output=result.to_dict(),
        evidence_ids_created=[e.evidence_id for e in new_evidence],
        state_before=state_before,
        state_after=state.to_dict(),
    )

    # --- deterministic assessment ---
    if not state.evidence_collected:
        _abstain(
            state,
            logger,
            missing="ingredient_or_allergen_evidence",
            reason="I do not have enough ingredient or allergen evidence to determine this.",
        )
        return _finish(state, logger, output_dir)

    state.status = RunStatus.READY_TO_ANSWER
    logger.log_step("status: ready_to_answer", state_after=state.to_dict())

    matches = find_milk_evidence(state.evidence_collected)
    state_before = state.to_dict()
    state.status = RunStatus.ANSWERED
    if matches:
        cited = "; ".join(f"'{m.value}' ({m.claim_type})" for m in matches)
        state.final_answer = f"Yes, evidence of a milk-derived ingredient was found: {cited}."
    else:
        state.final_answer = (
            "Based on the available ingredient and allergen evidence, no "
            "milk-derived ingredients were found. This reflects only the "
            "evidence collected and is not a guarantee the product is free of milk."
        )
    logger.log_step(
        "deterministic evidence assessment: answered",
        state_before=state_before,
        state_after=state.to_dict(),
    )

    return _finish(state, logger, output_dir)


def _abstain(state: AgentState, logger: TrajectoryLogger, *, missing: str, reason: str) -> None:
    state.missing_evidence.append(missing)
    state.status = RunStatus.INSUFFICIENT_EVIDENCE
    logger.log_step("status: insufficient_evidence", state_after=state.to_dict())

    state_before = state.to_dict()
    state.status = RunStatus.ABSTAINED
    state.final_answer = reason
    logger.log_step("abstained", state_before=state_before, state_after=state.to_dict())


def _finish(
    state: AgentState, logger: TrajectoryLogger, output_dir: Optional[Path]
) -> tuple[AgentState, Trajectory, Path]:
    trajectory = logger.finalize(state.status.value, state.final_answer)
    saved_path = logger.save(output_dir) if output_dir is not None else logger.save()
    return state, trajectory, saved_path

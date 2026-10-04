"""Run LabelLens v0 against its three fixture products and print a summary.

    python -m labellens.demo
"""

from __future__ import annotations

from labellens.agent.runtime import find_milk_evidence, run_query

QUERY = "Does this product contain milk?"

FIXTURE_PRODUCT_IDS = [
    "product_001",  # clear milk evidence -> ANSWERED (found)
    "product_002",  # no milk evidence -> ANSWERED (not found, hedged)
    "product_003",  # no ingredient data -> ABSTAINED
]


def print_case(product_identifier: str) -> None:
    state, trajectory, saved_path = run_query(QUERY, product_identifier)

    print("QUERY:")
    print(QUERY)
    print("PRODUCT:")
    print(state.resolved_product.name if state.resolved_product else "(not identified)")
    print("TOOLS USED:")
    for tool_name in state.tools_called:
        print(f"  {tool_name}")
    print("EVIDENCE:")
    if state.evidence_collected:
        highlighted = find_milk_evidence(state.evidence_collected) or state.evidence_collected
        for item in highlighted:
            print(f"  {item.value}")
            print(f"  Source: {item.source_name}")
    else:
        print("  (none collected)")
    print("RESULT:")
    print(f"  {state.final_answer}")
    print("STATUS:")
    print(f"  {state.status.value.upper()}")
    print("TRAJECTORY:")
    print(f"  saved to {saved_path}")
    print("-" * 60)


def main() -> None:
    for product_identifier in FIXTURE_PRODUCT_IDS:
        print_case(product_identifier)


if __name__ == "__main__":
    main()

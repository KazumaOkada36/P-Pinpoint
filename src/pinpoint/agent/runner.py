"""Claude API runner — handles tool-use conversation loop."""

from __future__ import annotations

import json
from typing import Any, Callable, Generator, Optional

import anthropic

from pinpoint.agent.system_prompt import SYSTEM_PROMPT

# --------------------------------------------------------------------------- #
# Tool schemas exposed to Claude                                               #
# --------------------------------------------------------------------------- #

TOOLS: list[dict] = [
    {
        "name": "parse_business_profile",
        "description": (
            "Parse a natural language business description into a structured profile "
            "that can be fed into the location recommender."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "business_type": {
                    "type": "string",
                    "description": "Type/industry of the business (e.g. 'SaaS startup', 'pharma', 'logistics')",
                },
                "employee_count": {
                    "type": "integer",
                    "description": "Number of employees (estimate if not stated)",
                },
                "valuation_usd": {
                    "type": "number",
                    "description": "Company valuation in USD (0 if unknown)",
                },
                "monthly_rent_budget_usd": {
                    "type": "number",
                    "description": "Monthly office rent budget in USD (0 if unknown)",
                },
                "priorities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "List of location priorities from: talent, cost_efficiency, gdp_growth, "
                        "manufacturing, tech_ecosystem, finance, healthcare, logistics, energy"
                    ),
                },
                "preferred_region": {
                    "type": "string",
                    "description": "Preferred US region if stated (e.g. 'West Coast', 'Southeast', 'any')",
                },
            },
            "required": ["business_type", "priorities"],
        },
    },
    {
        "name": "get_location_recommendations",
        "description": "Score and rank all US states for a given business profile. Returns top N states with scores and feature breakdown.",
        "input_schema": {
            "type": "object",
            "properties": {
                "business_type": {"type": "string"},
                "priorities": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "employee_count": {"type": "integer"},
                "monthly_rent_budget_usd": {"type": "number"},
                "preferred_region": {"type": "string"},
                "top_n": {
                    "type": "integer",
                    "description": "Number of top results to return (default 5)",
                },
            },
            "required": ["priorities"],
        },
    },
]


# --------------------------------------------------------------------------- #
# Runner                                                                       #
# --------------------------------------------------------------------------- #


class AgentRunner:
    """Drives a multi-turn Claude conversation with tool use."""

    def __init__(
        self,
        api_key: str,
        model: str,
        tool_handler: Callable[[str, dict], Any],
        on_text: Optional[Callable[[str], None]] = None,
        on_thinking: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._tool_handler = tool_handler
        self._on_text = on_text or (lambda t: None)
        self._on_thinking = on_thinking or (lambda t: None)
        self._history: list[dict] = []

    # ------------------------------------------------------------------
    def send(self, user_message: str) -> str:
        """Send a user message, run tool loop, return final assistant text."""
        self._history.append({"role": "user", "content": user_message})
        return self._run_loop()

    def reset(self) -> None:
        self._history.clear()

    # ------------------------------------------------------------------
    def _run_loop(self) -> str:
        """Agentic tool-use loop — keeps calling Claude until no more tool calls."""
        while True:
            self._on_thinking("Thinking")
            response = self._client.messages.create(
                model=self._model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=self._history,
            )

            # Collect assistant message content
            assistant_content = response.content
            self._history.append({"role": "assistant", "content": assistant_content})

            # Check stop reason
            if response.stop_reason == "end_turn":
                # Extract plain text
                text = " ".join(
                    block.text for block in assistant_content if hasattr(block, "text")
                )
                self._on_text(text)
                return text

            if response.stop_reason == "tool_use":
                # Execute all tool calls
                tool_results = []
                for block in assistant_content:
                    if block.type == "tool_use":
                        self._on_thinking(f"Running {block.name}")
                        try:
                            result = self._tool_handler(block.name, block.input)
                            tool_results.append(
                                {
                                    "type": "tool_result",
                                    "tool_use_id": block.id,
                                    "content": json.dumps(result),
                                }
                            )
                        except Exception as e:
                            tool_results.append(
                                {
                                    "type": "tool_result",
                                    "tool_use_id": block.id,
                                    "is_error": True,
                                    "content": str(e),
                                }
                            )

                self._history.append({"role": "user", "content": tool_results})
                continue  # loop back to call Claude again

            # Unexpected stop reason — return whatever text we have
            text = " ".join(
                block.text for block in assistant_content if hasattr(block, "text")
            )
            return text

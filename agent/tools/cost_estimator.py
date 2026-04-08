"""Cost estimator tool - estimates token usage and compute cost before building."""

import os
import json
from langchain_core.tools import tool


# Rough token-per-step estimates based on observed ModelForge builds
PHASE_TOKEN_ESTIMATES = {
    "data_analysis": {"input": 2000, "output": 1500},
    "research": {"input": 3000, "output": 2000},
    "environment_setup": {"input": 4000, "output": 3000},
    "model_training": {"input": 8000, "output": 6000},
    "hyperparameter_tuning": {"input": 5000, "output": 4000},
    "delivery": {"input": 3000, "output": 2500},
    "api_generation": {"input": 2000, "output": 1500},
    "model_card": {"input": 1500, "output": 1000},
    "experiment_logging": {"input": 1000, "output": 500},
    "fairness_check": {"input": 3000, "output": 2000},
}

# Claude pricing per million tokens (as of 2025)
PRICING = {
    "claude-sonnet": {"input": 3.0, "output": 15.0},
    "claude-sonnet-4-5-20250514": {"input": 3.0, "output": 15.0},
    "claude-opus": {"input": 15.0, "output": 75.0},
    "claude-haiku": {"input": 0.25, "output": 1.25},
    "sonnet": {"input": 3.0, "output": 15.0},
    "opus": {"input": 15.0, "output": 75.0},
    "haiku": {"input": 0.25, "output": 1.25},
}


def estimate_build_cost(plan_text: str, model_name: str = "claude-sonnet") -> dict:
    """Estimate token usage and cost for a build plan.

    Args:
        plan_text: The plan text from the planning agent.
        model_name: The model being used.

    Returns:
        Dict with estimated tokens, cost, and breakdown.
    """
    # Count phases mentioned in the plan
    plan_lower = plan_text.lower()
    phases = []

    # Always include core phases
    phases.append("data_analysis")
    phases.append("research")
    phases.append("environment_setup")
    phases.append("model_training")

    # Count models to try
    model_count = plan_lower.count("- ") if "models to try" in plan_lower else 2
    model_count = max(2, min(model_count, 6))

    # Conditional phases
    if "tun" in plan_lower or "hyperparameter" in plan_lower or "grid" in plan_lower:
        phases.append("hyperparameter_tuning")

    phases.append("delivery")

    if "api" in plan_lower or "endpoint" in plan_lower or "serve" in plan_lower:
        phases.append("api_generation")

    if "model card" in plan_lower or "documentation" in plan_lower:
        phases.append("model_card")

    if "experiment" in plan_lower or "mlflow" in plan_lower or "track" in plan_lower:
        phases.append("experiment_logging")

    if "fairness" in plan_lower or "bias" in plan_lower:
        phases.append("fairness_check")

    # Calculate estimates
    total_input = 0
    total_output = 0
    breakdown = {}

    for phase in phases:
        est = PHASE_TOKEN_ESTIMATES.get(phase, {"input": 2000, "output": 1500})
        # Scale model training by number of models
        if phase == "model_training":
            est = {"input": est["input"] * model_count, "output": est["output"] * model_count}
        total_input += est["input"]
        total_output += est["output"]
        breakdown[phase] = est

    # Add overhead for agent reasoning between steps (roughly 30%)
    overhead_input = int(total_input * 0.3)
    overhead_output = int(total_output * 0.3)
    total_input += overhead_input
    total_output += overhead_output

    # Get pricing
    pricing = PRICING.get(model_name, PRICING.get("claude-sonnet"))
    cost_input = (total_input / 1_000_000) * pricing["input"]
    cost_output = (total_output / 1_000_000) * pricing["output"]
    total_cost = cost_input + cost_output

    return {
        "estimated_input_tokens": total_input,
        "estimated_output_tokens": total_output,
        "estimated_total_tokens": total_input + total_output,
        "estimated_cost_usd": round(total_cost, 4),
        "model": model_name,
        "phases": phases,
        "model_count": model_count,
        "breakdown": breakdown,
    }


def format_cost_estimate(estimate: dict) -> str:
    """Format cost estimate as a human-readable string."""
    lines = []
    lines.append("**Estimated Build Cost**")
    lines.append(f"  Tokens: ~{estimate['estimated_total_tokens']:,} ({estimate['estimated_input_tokens']:,} in / {estimate['estimated_output_tokens']:,} out)")
    lines.append(f"  Cost: ~${estimate['estimated_cost_usd']:.4f} USD")
    lines.append(f"  Model: {estimate['model']}")
    lines.append(f"  Phases: {len(estimate['phases'])} ({', '.join(estimate['phases'])})")
    lines.append(f"  Models to train: {estimate['model_count']}")
    lines.append("")
    lines.append("  *Estimates are rough approximations. Actual usage depends on dataset complexity and agent decisions.*")
    return "\n".join(lines)

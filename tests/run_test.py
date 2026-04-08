"""Generic test runner for ModelForge - invokes the builder agent programmatically."""

import sys
import os
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.ml_agent import create_agent


def run_test(prompt: str, test_name: str):
    """Run a ModelForge test and report results."""
    print(f"=== TEST: {test_name} ===")
    print(f"Prompt: {prompt[:100]}...")
    print()

    agent = create_agent()
    start = time.time()

    try:
        result = agent.invoke(
            {"input": prompt, "chat_history": []},
        )

        elapsed = time.time() - start
        output = result.get("output", "")
        if isinstance(output, list):
            texts = []
            for item in output:
                if isinstance(item, dict) and item.get("type") == "text":
                    texts.append(item["text"])
                elif isinstance(item, str):
                    texts.append(item)
            output = "\n".join(texts)

        steps = result.get("intermediate_steps", [])

        print(f"\n{'='*60}")
        print(f"TEST: {test_name}")
        print(f"Status: COMPLETED")
        print(f"Steps: {len(steps)}")
        print(f"Time: {elapsed:.0f}s")
        print(f"{'='*60}")
        print(f"\nAgent output:\n{output}")

    except Exception as e:
        elapsed = time.time() - start
        print(f"\n{'='*60}")
        print(f"TEST: {test_name}")
        print(f"Status: FAILED")
        print(f"Error: {e}")
        print(f"Time: {elapsed:.0f}s")
        print(f"{'='*60}")


if __name__ == "__main__":
    test_name = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    prompt = sys.argv[2] if len(sys.argv) > 2 else ""
    if not prompt:
        print("Usage: python run_test.py <test_name> <prompt>")
        sys.exit(1)
    run_test(prompt, test_name)

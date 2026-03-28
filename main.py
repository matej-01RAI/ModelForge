#!/usr/bin/env python3
"""ModelForge - Terminal Chat Interface.

An autonomous AI agent that builds machine learning models.
Supports multiple providers: Anthropic API, Azure AI Foundry, Claude Code CLI.
"""

import os
import sys
import signal
import threading
import time
import itertools
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.theme import Theme

from langchain_core.callbacks import BaseCallbackHandler

from agent.ml_agent import create_agent
from agent.planning_agent import create_planning_agent
from agent.llm_factory import create_llm
import config

custom_theme = Theme({
    "user": "bold cyan",
    "agent": "bold green",
    "system": "bold yellow",
    "error": "bold red",
    "status": "dim cyan",
    "phase": "bold magenta",
    "tool_label": "bold cyan",
    "tool_detail": "cyan",
    "thinking": "dim italic",
    "timer": "dim",
})

console = Console(theme=custom_theme)


WELCOME_BANNER = """
# ModelForge

An autonomous AI agent that builds the best possible ML models for your data.

**How it works:**
1. Describe your task - the agent will ask clarifying questions if needed
2. Review the plan - the agent proposes an approach before building
3. Watch it work - live status updates show what's happening
4. Get results - trained model, metrics, plots, and code

**Commands:**
- `/quit` or `/exit` to exit
- `/clear` to clear conversation history
- `/workspace` to show workspace directory
- `/tokens` to show session token usage
- `/build` to skip planning and go straight to building

**Example prompts:**
- "I have a CSV at /path/to/data.csv, target column is 'label', build me a classifier"
- "Build a regression model to predict house prices from housing.csv"
- "I need a time series forecasting model for stock data"
"""


# ── Spinner Thread ────────────────────────────────────────────────────────

class ThinkingSpinner:
    """A simple animated spinner that shows the agent is thinking."""

    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self):
        self._stop = threading.Event()
        self._thread = None
        self._status = "Thinking..."
        self._lock = threading.Lock()
        self.start_time = 0

    def _run(self):
        frames = itertools.cycle(self.FRAMES)
        while not self._stop.is_set():
            frame = next(frames)
            elapsed = time.time() - self.start_time
            mins, secs = divmod(int(elapsed), 60)
            time_str = f"{mins}:{secs:02d}" if mins else f"{secs}s"
            with self._lock:
                status = self._status
            # \r + clear line + write status
            line = f"\r\033[K  {frame} \033[36m{status}\033[0m  \033[2m[{time_str}]\033[0m"
            sys.stderr.write(line)
            sys.stderr.flush()
            self._stop.wait(0.1)
        # Clear the spinner line when done
        sys.stderr.write("\r\033[K")
        sys.stderr.flush()

    def start(self, status="Thinking..."):
        self._stop.clear()
        self._status = status
        self.start_time = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def update(self, status):
        with self._lock:
            self._status = status

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
            self._thread = None


# ── Live Status Callback Handler ──────────────────────────────────────────

class LiveStatusCallback(BaseCallbackHandler):
    """Callback that prints persistent status lines + animated spinner + token tracking."""

    TOOL_ICONS = {
        "run_terminal_command": ">>",
        "read_file": "<<",
        "write_file": "=>",
        "list_directory": "[]",
        "search_web": "??",
        "fetch_url": "<>",
        "analyze_dataset": "##",
    }

    TOOL_VERBS = {
        "run_terminal_command": "Running command",
        "read_file": "Reading",
        "write_file": "Writing",
        "list_directory": "Listing",
        "search_web": "Searching",
        "fetch_url": "Fetching",
        "analyze_dataset": "Analyzing",
    }

    def __init__(self):
        self.tool_calls = 0
        self.start_time = time.time()
        self._lock = threading.Lock()
        self._spinner = ThinkingSpinner()
        # Token tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        self.llm_calls = 0
        self.cache_read_tokens = 0
        self.cache_creation_tokens = 0

    def _elapsed(self):
        return time.time() - self.start_time

    @staticmethod
    def _shorten(text, max_len=50):
        """Truncate text with ellipsis if too long."""
        if len(text) > max_len:
            return text[:max_len - 3] + "..."
        return text

    @staticmethod
    def _short_path(path):
        """Show just the last 2 path components for readability."""
        if not path:
            return ""
        parts = path.replace("\\", "/").rstrip("/").split("/")
        return "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]

    def _format_tool_detail(self, tool_name, input_str):
        """Create a human-readable description of what the tool is doing."""
        verb = self.TOOL_VERBS.get(tool_name, tool_name)

        # LangChain passes input_str as a string in tool-calling agents — parse it
        if isinstance(input_str, str):
            try:
                import ast
                input_str = ast.literal_eval(input_str)
            except (ValueError, SyntaxError):
                return verb

        if not isinstance(input_str, dict):
            return verb

        if tool_name == "run_terminal_command":
            cmd = input_str.get("command", "")
            if "pip install" in cmd:
                pkgs = cmd.split("pip install")[-1].strip().split("&&")[0].strip()
                # Show just package names, not flags
                pkg_names = [p for p in pkgs.split() if not p.startswith("-")]
                return f"Installing {self._shorten(' '.join(pkg_names), 45)}"
            elif "python" in cmd and ".py" in cmd:
                parts = cmd.split()
                for p in parts:
                    if p.endswith(".py"):
                        return f"Running {os.path.basename(p)}"
                return "Running Python script"
            elif "mkdir" in cmd:
                return "Creating directories"
            elif "venv" in cmd or "virtualenv" in cmd:
                return "Setting up virtual environment"
            elif "cp " in cmd or "mv " in cmd:
                return "Copying files" if "cp " in cmd else "Moving files"
            elif "pip" in cmd and "upgrade" in cmd:
                return "Upgrading pip"
            elif "git " in cmd:
                sub = cmd.split("git")[-1].strip().split()[0] if "git" in cmd else ""
                return f"Git {sub}"
            elif "cd " in cmd and "&&" in cmd:
                # Show the actual command after cd
                after_cd = cmd.split("&&", 1)[-1].strip()
                return f"$ {self._shorten(after_cd, 50)}"
            else:
                return f"$ {self._shorten(cmd, 50)}"

        elif tool_name == "write_file":
            path = input_str.get("file_path", "")
            return f"Writing {self._short_path(path)}"

        elif tool_name == "read_file":
            path = input_str.get("file_path", "")
            return f"Reading {self._short_path(path)}"

        elif tool_name == "analyze_dataset":
            path = input_str.get("file_path", "")
            name = self._short_path(path)
            return f"Analyzing dataset {name}" if name else "Analyzing dataset"

        elif tool_name == "search_web":
            query = input_str.get("query", "")
            return f'Searching "{self._shorten(query, 45)}"'

        elif tool_name == "list_directory":
            path = input_str.get("dir_path", "")
            name = self._short_path(path)
            return f"Listing {name}" if name else "Listing directory"

        elif tool_name == "fetch_url":
            url = input_str.get("url", "")
            # Strip protocol for brevity
            short_url = url.replace("https://", "").replace("http://", "")
            return f"Fetching {self._shorten(short_url, 45)}"

        return verb

    def start(self):
        """Start tracking and show initial spinner."""
        self.start_time = time.time()
        self.tool_calls = 0
        self._spinner.start("Thinking...")
        console.print()  # blank line before status output

    def stop(self):
        """Stop the spinner."""
        self._spinner.stop()

    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name", "unknown")
        with self._lock:
            self.tool_calls += 1
            detail = self._format_tool_detail(tool_name, input_str)
            icon = self.TOOL_ICONS.get(tool_name, "--")

            # Stop spinner, print a permanent line, restart spinner
            self._spinner.stop()

            elapsed = self._elapsed()
            mins, secs = divmod(int(elapsed), 60)
            time_str = f"{mins}:{secs:02d}" if mins else f"0:{secs:02d}"

            console.print(
                f"  [timer][{time_str}][/timer] "
                f"[tool_label]{icon}[/tool_label] "
                f"[tool_detail]{detail}[/tool_detail]"
            )

            self._spinner.start(f"Working on step {self.tool_calls}...")

    def on_tool_end(self, output, **kwargs):
        with self._lock:
            self._spinner.update("Thinking about next step...")

    def on_llm_start(self, serialized, prompts, **kwargs):
        with self._lock:
            if self.tool_calls == 0:
                self._spinner.update("Analyzing your request...")
            else:
                self._spinner.update("Deciding next step...")

    def on_llm_end(self, response, **kwargs):
        """Track token usage from each LLM call."""
        with self._lock:
            self.llm_calls += 1
            for gen_list in (response.generations or []):
                for gen in gen_list:
                    msg = getattr(gen, "message", None)
                    if not msg:
                        continue
                    usage = getattr(msg, "usage_metadata", None)
                    if usage:
                        self.total_input_tokens += usage.get("input_tokens", 0)
                        self.total_output_tokens += usage.get("output_tokens", 0)
                        self.total_tokens += usage.get("total_tokens", 0)
                        details = usage.get("input_token_details", {})
                        self.cache_read_tokens += details.get("cache_read", 0)
                        self.cache_creation_tokens += details.get("cache_creation", 0)

    def on_agent_action(self, action, **kwargs):
        pass

    def format_token_summary(self):
        """Format a human-readable token usage summary."""
        parts = []
        parts.append(f"tokens: {self.total_tokens:,} total")
        parts.append(f"{self.total_input_tokens:,} in")
        parts.append(f"{self.total_output_tokens:,} out")
        if self.cache_read_tokens:
            parts.append(f"{self.cache_read_tokens:,} cached")
        parts.append(f"{self.llm_calls} LLM calls")
        return " | ".join(parts)


# ── Chat State ────────────────────────────────────────────────────────────

class TokenTracker:
    """Tracks cumulative token usage across the entire session."""

    def __init__(self):
        self.total_input = 0
        self.total_output = 0
        self.total = 0
        self.cache_read = 0
        self.cache_creation = 0
        self.llm_calls = 0
        self.turns = 0

    def add_from_callback(self, callback):
        """Accumulate tokens from a LiveStatusCallback."""
        self.total_input += callback.total_input_tokens
        self.total_output += callback.total_output_tokens
        self.total += callback.total_tokens
        self.cache_read += callback.cache_read_tokens
        self.cache_creation += callback.cache_creation_tokens
        self.llm_calls += callback.llm_calls
        self.turns += 1

    def add_from_llm_response(self, response):
        """Accumulate tokens from a direct LLM response (planner)."""
        usage = getattr(response, "usage_metadata", None)
        if usage:
            self.total_input += usage.get("input_tokens", 0)
            self.total_output += usage.get("output_tokens", 0)
            self.total += usage.get("total_tokens", 0)
            details = usage.get("input_token_details", {})
            self.cache_read += details.get("cache_read", 0)
            self.cache_creation += details.get("cache_creation", 0)
        self.llm_calls += 1
        self.turns += 1

    def display(self):
        """Print a formatted token usage summary to the console."""
        console.print()
        console.print(Panel(
            f"[bold]Session Token Usage[/bold]\n\n"
            f"  Total tokens:    [cyan]{self.total:,}[/cyan]\n"
            f"  Input tokens:    {self.total_input:,}\n"
            f"  Output tokens:   {self.total_output:,}\n"
            f"  Cache read:      {self.cache_read:,}\n"
            f"  Cache creation:  {self.cache_creation:,}\n"
            f"  LLM calls:       {self.llm_calls}\n"
            f"  Turns:           {self.turns}",
            border_style="cyan",
            title="Token Usage",
            title_align="left",
        ))


class ChatState:
    CHATTING = "chatting"
    PLAN_READY = "plan_ready"
    BUILDING = "building"

    def __init__(self):
        self.phase = self.CHATTING
        self.chat_history = []
        self.current_plan = None
        self.tokens = TokenTracker()


# ── Signal Handler ────────────────────────────────────────────────────────

_active_callback = None

def handle_sigint(sig, frame):
    if _active_callback:
        _active_callback.stop()
    console.print("\n\n[system]Interrupted. Type /quit to exit.[/system]")


# ── Helpers ───────────────────────────────────────────────────────────────

_intent_llm = None

def _get_intent_llm():
    """Lazy-init a lightweight LLM for intent classification."""
    global _intent_llm
    if _intent_llm is None:
        _intent_llm = create_llm(temperature=0.0, max_tokens=16)
    return _intent_llm


def classify_plan_intent(user_input: str) -> str:
    """Use the LLM to classify whether the user approves, rejects, or modifies the plan.

    Returns one of: "approve", "reject", "modify"
    """
    # Fast-path: very short obvious inputs (avoid LLM call for "yes"/"y"/"go")
    normalized = user_input.lower().strip().rstrip(".!,")
    if normalized in ("yes", "y", "go", "ok", "sure", "yep", "yeah", "lgtm"):
        return "approve"
    if normalized in ("no", "n", "stop", "cancel", "nope", "abort"):
        return "reject"

    prompt = (
        "You are an intent classifier. The user was shown a plan and asked to approve it.\n"
        "Classify their response as exactly one of these words:\n"
        "- approve (they want to proceed with the plan as-is)\n"
        "- modify (they want to change something about the plan)\n"
        "- reject (they want to cancel or start over)\n\n"
        f'User response: "{user_input}"\n\n'
        "Classification:"
    )

    try:
        llm = _get_intent_llm()
        response = llm.invoke(prompt)
        result = response.content.strip().lower().split()[0].rstrip(".,!\"'")
        if result in ("approve", "modify", "reject"):
            return result
        # If the LLM returns something unexpected, default to approve for positive-sounding text
        return "modify"
    except Exception:
        # Fallback: if LLM call fails, default to modify (safest — sends to planner)
        return "modify"


def run_planning_turn(planner, user_input, chat_history):
    """Run planner and return the full response object (for token tracking)."""
    response = planner.invoke({
        "input": user_input,
        "chat_history": chat_history,
    })
    return response


def run_builder(executor, task_input, chat_history, callback):
    callback.start()
    try:
        result = executor.invoke(
            {
                "input": task_input,
                "chat_history": chat_history,
            },
            config={"callbacks": [callback]},
        )
        return result
    finally:
        callback.stop()


def extract_output_text(output):
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        texts = []
        for item in output:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(item["text"])
            elif isinstance(item, str):
                texts.append(item)
        return "\n".join(texts) if texts else str(output)
    return str(output)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    global _active_callback
    signal.signal(signal.SIGINT, handle_sigint)

    console.print(Panel(Markdown(WELCOME_BANNER), border_style="green", title="ModelForge"))

    from agent.llm_factory import get_provider_display_name

    console.print(f"[system]Workspace: {config.WORKSPACE_DIR}[/system]")
    console.print(f"[system]Provider: {get_provider_display_name()}[/system]")
    console.print()

    try:
        planner = create_planning_agent()
        executor = create_agent()
        console.print("[system]Agents initialized (planner + builder).[/system]\n")
    except Exception as e:
        console.print(f"[error]Failed to initialize: {e}[/error]")
        sys.exit(1)

    state = ChatState()

    while True:
        try:
            console.print("[user]You:[/user] ", end="")
            user_input = input().strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[system]Goodbye![/system]")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit"):
            console.print("[system]Goodbye![/system]")
            break
        elif user_input.lower() == "/clear":
            state = ChatState()
            console.print("[system]Conversation cleared. Starting fresh.[/system]\n")
            continue
        elif user_input.lower() == "/workspace":
            console.print(f"[system]Workspace: {config.WORKSPACE_DIR}[/system]\n")
            continue
        elif user_input.lower() == "/tokens":
            state.tokens.display()
            console.print()
            continue
        elif user_input.lower() == "/build" and state.current_plan:
            user_input = "yes, proceed"

        console.print()

        # ── Planning / Chatting ───────────────────────────────────────
        if state.phase in (ChatState.CHATTING, ChatState.PLAN_READY):
            plan_approved = False
            if state.phase == ChatState.PLAN_READY:
                intent = classify_plan_intent(user_input)
                if intent == "approve":
                    plan_approved = True
                elif intent == "reject":
                    state.phase = ChatState.CHATTING
                    state.current_plan = None
                    console.print("[system]Plan cancelled. Describe what you'd like instead.[/system]\n")
                    continue
                # intent == "modify" falls through to planner

            if plan_approved:
                state.phase = ChatState.BUILDING
                console.print(Panel(
                    "[phase]Starting build...[/phase]",
                    border_style="magenta",
                    title="Builder Agent",
                    title_align="left",
                ))

                callback = LiveStatusCallback()
                _active_callback = callback

                try:
                    build_input = f"PLAN:\n{state.current_plan}\n\nPlease execute this plan now."
                    result = run_builder(executor, build_input, state.chat_history, callback)
                    state.tokens.add_from_callback(callback)

                    output = extract_output_text(result.get("output", ""))
                    steps = result.get("intermediate_steps", [])

                    # Summary line
                    elapsed = callback._elapsed()
                    mins, secs = divmod(int(elapsed), 60)
                    time_str = f"{mins}m {secs}s" if mins else f"{secs}s"
                    console.print()
                    console.print(f"  [system]Done: {len(steps)} steps in {time_str} | {callback.format_token_summary()}[/system]")
                    console.print()

                    console.print(Panel(
                        Markdown(output),
                        border_style="green",
                        title="Agent",
                        title_align="left",
                    ))

                    state.chat_history.append({"role": "user", "content": build_input})
                    state.chat_history.append({"role": "assistant", "content": output})
                    state.phase = ChatState.CHATTING
                    state.current_plan = None

                except KeyboardInterrupt:
                    callback.stop()
                    console.print("\n[system]Build interrupted. You can try again or adjust the plan.[/system]")
                    state.phase = ChatState.CHATTING
                except Exception as e:
                    callback.stop()
                    console.print(f"[error]Build error: {e}[/error]")
                    state.phase = ChatState.CHATTING
                finally:
                    _active_callback = None

            else:
                # Planning agent turn
                spinner = ThinkingSpinner()
                try:
                    spinner.start("Planning agent thinking...")
                    response = run_planning_turn(planner, user_input, state.chat_history)
                    spinner.stop()

                    state.tokens.add_from_llm_response(response)
                    response_text = extract_output_text(response.content)
                    has_plan = "PLAN:" in response_text

                    console.print(Panel(
                        Markdown(response_text),
                        border_style="blue" if not has_plan else "magenta",
                        title="Planner" if not has_plan else "Proposed Plan",
                        title_align="left",
                    ))

                    state.chat_history.append({"role": "user", "content": user_input})
                    state.chat_history.append({"role": "assistant", "content": response_text})

                    if has_plan:
                        state.current_plan = response_text
                        state.phase = ChatState.PLAN_READY
                        console.print("[system]Review the plan above. Type 'yes' to start building, or suggest changes.[/system]")
                    else:
                        state.phase = ChatState.CHATTING

                except KeyboardInterrupt:
                    spinner.stop()
                    console.print("\n[system]Interrupted.[/system]")
                except Exception as e:
                    spinner.stop()
                    console.print(f"[error]Planner error: {e}[/error]")

        # ── Building Phase (follow-ups after build) ───────────────────
        elif state.phase == ChatState.BUILDING:
            callback = LiveStatusCallback()
            _active_callback = callback
            try:
                result = run_builder(executor, user_input, state.chat_history, callback)
                state.tokens.add_from_callback(callback)

                output = extract_output_text(result.get("output", ""))
                steps = result.get("intermediate_steps", [])

                if steps:
                    elapsed = callback._elapsed()
                    mins, secs = divmod(int(elapsed), 60)
                    time_str = f"{mins}m {secs}s" if mins else f"{secs}s"
                    console.print()
                    console.print(f"  [system]Done: {len(steps)} steps in {time_str} | {callback.format_token_summary()}[/system]")

                console.print()
                console.print(Panel(
                    Markdown(output),
                    border_style="green",
                    title="Agent",
                    title_align="left",
                ))

                state.chat_history.append({"role": "user", "content": user_input})
                state.chat_history.append({"role": "assistant", "content": output})

            except KeyboardInterrupt:
                callback.stop()
                console.print("\n[system]Interrupted.[/system]")
            except Exception as e:
                callback.stop()
                console.print(f"[error]Agent error: {e}[/error]")
            finally:
                _active_callback = None

        console.print()


if __name__ == "__main__":
    main()

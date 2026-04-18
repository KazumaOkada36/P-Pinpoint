"""Interactive REPL for Pinpoint — prompt_toolkit based."""

from __future__ import annotations

import sys
from typing import Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich import box

from pinpoint.agent.config import Config, HISTORY_FILE, MODELS
from pinpoint.ui.status import ThinkingStatus

# --------------------------------------------------------------------------- #
# Pixel-art splash screen                                                      #
# --------------------------------------------------------------------------- #

_LOGO = r"""
  ██████╗ ██╗███╗  ██╗██████╗  ██████╗ ██╗███╗  ██╗████████╗
  ██╔══██╗██║████╗ ██║██╔══██╗██╔═══██╗██║████╗ ██║╚══██╔══╝
  ██████╔╝██║██╔██╗██║██████╔╝██║   ██║██║██╔██╗██║   ██║
  ██╔═══╝ ██║██║╚████║██╔═══╝ ██║   ██║██║██║╚████║   ██║
  ██║     ██║██║ ╚███║██║     ╚██████╔╝██║██║ ╚███║   ██║
  ╚═╝     ╚═╝╚═╝  ╚══╝╚═╝      ╚═════╝ ╚═╝╚═╝  ╚══╝   ╚═╝
"""

# --------------------------------------------------------------------------- #
# Slash commands                                                               #
# --------------------------------------------------------------------------- #

SLASH_COMMANDS: dict[str, str] = {
    "/help": "Show command reference",
    "/model": "Switch the Claude model",
    "/config": "Show active configuration",
    "/clear": "Clear conversation history",
    "/train": "Retrain the recommender model",
    "/import": "Import custom CSV data",
    "/top": "Change number of recommendations returned",
    "/exit": "Exit pinpoint",
}

# Placeholder suggestions shown in the prompt
_SUGGESTIONS = [
    "We're a 50-person SaaS startup looking for a tech hub on the West Coast...",
    "Pharma company, 200 employees, need proximity to research universities...",
    "E-commerce logistics startup, budget-conscious, needs good infrastructure...",
    "FinTech firm, Series B, prioritize talent and financial ecosystem...",
    "Healthcare AI company looking for a state with strong biotech presence...",
]

# --------------------------------------------------------------------------- #
# Prompt toolkit style                                                         #
# --------------------------------------------------------------------------- #

_PROMPT_STYLE = Style.from_dict(
    {
        "prompt": "#00d4ff bold",
        "bottom-toolbar": "bg:#1a1a2e #555577",
        "completion-menu.completion": "bg:#1a1a2e #00d4ff",
        "completion-menu.completion.current": "bg:#00d4ff #1a1a2e bold",
        "auto-suggestion": "#3a3a5c",
    }
)


# --------------------------------------------------------------------------- #
# Tool handler — bridges Claude tool calls to the recommender                 #
# --------------------------------------------------------------------------- #

def make_tool_handler(config: Config):
    """Return a closure that handles Claude tool calls."""
    from pinpoint.recommender.trainer import load_or_train

    _model_cache: dict = {}

    def _get_model():
        if "model" not in _model_cache:
            _model_cache["model"] = load_or_train(data_dir=config.data_dir, quiet=True)
        return _model_cache["model"]

    def handler(tool_name: str, tool_input: dict) -> dict:
        if tool_name == "parse_business_profile":
            # Just echo back — Claude already filled the structured fields
            return {"status": "ok", "profile": tool_input}

        if tool_name == "get_location_recommendations":
            model = _get_model()
            priorities = tool_input.get("priorities", [])
            top_n = int(tool_input.get("top_n", config.top_n))
            preferred_region = tool_input.get("preferred_region")

            results = model.recommend(priorities, top_n=top_n, preferred_region=preferred_region)

            return {
                "recommendations": [
                    {
                        "rank": r.rank,
                        "state": r.state,
                        "state_name": r.state_name,
                        "score": r.score,
                        "top_features": [
                            {"feature": f, "value": round(v, 4)}
                            for f, v in r.top_features
                        ],
                    }
                    for r in results
                ]
            }

        return {"error": f"Unknown tool: {tool_name}"}

    return handler


# --------------------------------------------------------------------------- #
# Main REPL                                                                    #
# --------------------------------------------------------------------------- #

class InteractiveTerminal:
    def __init__(self, config: Config) -> None:
        self._config = config
        self._console = Console()
        self._status = ThinkingStatus(self._console)
        self._runner: Optional[object] = None  # AgentRunner, created lazily
        self._ctrl_c_count = 0

    # ------------------------------------------------------------------
    def run(self) -> None:
        self._print_splash()
        self._ensure_runner()

        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        session: PromptSession = PromptSession(
            history=FileHistory(str(HISTORY_FILE)),
            auto_suggest=AutoSuggestFromHistory(),
            completer=WordCompleter(list(SLASH_COMMANDS.keys()), sentence=True),
            style=_PROMPT_STYLE,
            bottom_toolbar=self._toolbar,
            key_bindings=self._make_bindings(),
        )

        import itertools
        suggestion_cycle = itertools.cycle(_SUGGESTIONS)

        while True:
            try:
                placeholder = HTML(
                    f'<style color="#3a3a5c">{next(suggestion_cycle)}</style>'
                )
                user_input = session.prompt(
                    HTML('<ansi_bright_cyan><b>❯ </b></ansi_bright_cyan>'),
                    placeholder=placeholder,
                ).strip()
            except KeyboardInterrupt:
                self._ctrl_c_count += 1
                if self._ctrl_c_count >= 2:
                    self._console.print("\n[dim]Goodbye.[/dim]")
                    sys.exit(0)
                self._console.print(
                    "[dim]  Press Ctrl+C again to exit, or type /exit[/dim]"
                )
                continue
            except EOFError:
                self._console.print("\n[dim]Goodbye.[/dim]")
                break

            self._ctrl_c_count = 0

            if not user_input:
                continue

            if user_input.startswith("/"):
                self._handle_slash(user_input)
                continue

            self._chat(user_input)

    # ------------------------------------------------------------------
    def _chat(self, message: str) -> None:
        if not self._runner:
            self._console.print("[red]No API key configured. Run `pinpoint setup`.[/red]")
            return

        self._status.start("Thinking")
        try:
            reply = self._runner.send(message)
        except Exception as e:
            self._status.stop()
            self._console.print(f"[red]Error:[/red] {e}")
            return
        self._status.stop()

        self._console.print()
        self._console.print(Markdown(reply))
        self._console.print()

    # ------------------------------------------------------------------
    def _handle_slash(self, cmd: str) -> None:
        parts = cmd.split(None, 1)
        command = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if command == "/help":
            self._print_help()
        elif command == "/model":
            self._pick_model()
        elif command == "/config":
            self._show_config()
        elif command == "/clear":
            if self._runner:
                self._runner.reset()
            self._console.print("[green]Conversation cleared.[/green]")
        elif command == "/train":
            self._retrain()
        elif command == "/import":
            if arg:
                self._import_data(arg.strip())
            else:
                self._console.print("[yellow]Usage:[/yellow] /import <path/to/data.csv>")
        elif command == "/top":
            if arg.isdigit():
                self._config.top_n = int(arg)
                self._config.save()
                self._console.print(f"[green]Top-N set to {self._config.top_n}[/green]")
            else:
                self._console.print("[yellow]Usage:[/yellow] /top <number>")
        elif command == "/exit":
            self._console.print("[dim]Goodbye.[/dim]")
            sys.exit(0)
        else:
            self._console.print(f"[yellow]Unknown command:[/yellow] {command}  (try /help)")

    # ------------------------------------------------------------------
    def _ensure_runner(self) -> None:
        if not self._config.api_key:
            self._console.print(
                Panel(
                    "[yellow]No API key set.[/yellow]\n"
                    "Run [cyan]pinpoint setup[/cyan] to configure your Anthropic API key,\n"
                    "or set the [cyan]ANTHROPIC_API_KEY[/cyan] environment variable.",
                    title="Setup Required",
                    border_style="yellow",
                )
            )
            return
        self._build_runner()

    def _build_runner(self) -> None:
        from pinpoint.agent.runner import AgentRunner

        def on_thinking(label: str):
            self._status.update(label)

        self._runner = AgentRunner(
            api_key=self._config.api_key,
            model=self._config.model,
            tool_handler=make_tool_handler(self._config),
            on_thinking=on_thinking,
        )

    # ------------------------------------------------------------------
    def _print_splash(self) -> None:
        self._console.print(
            f"[bold #00d4ff]{_LOGO}[/bold #00d4ff]"
        )
        self._console.print(
            "  [dim]AI-powered business location recommender[/dim]"
        )
        self._console.print(
            f"  [dim]v0.1.0  ·  model: {self._config.model}[/dim]\n"
        )
        self._console.print(
            "  [dim]Describe your business, or type [cyan]/help[/cyan] for commands.[/dim]\n"
        )

    def _print_help(self) -> None:
        table = Table(box=box.SIMPLE_HEAVY, show_header=False, padding=(0, 2))
        table.add_column("cmd", style="cyan")
        table.add_column("desc", style="dim")
        for cmd, desc in SLASH_COMMANDS.items():
            table.add_row(cmd, desc)
        self._console.print(table)

    def _show_config(self) -> None:
        table = Table(box=box.SIMPLE_HEAVY, show_header=False, padding=(0, 2))
        table.add_column("key", style="cyan")
        table.add_column("value")
        table.add_row("api_key", self._config.masked_key)
        table.add_row("model", self._config.model)
        table.add_row("top_n", str(self._config.top_n))
        table.add_row("data_dir", str(self._config.data_dir or "(bundled)"))
        self._console.print(table)

    def _pick_model(self) -> None:
        self._console.print("\n[bold]Select model:[/bold]")
        options = list(MODELS.items())
        for i, (mid, label) in enumerate(options, 1):
            marker = "[bold cyan]*[/bold cyan]" if mid == self._config.model else " "
            self._console.print(f"  {marker} [{i}] {label}  [dim]{mid}[/dim]")

        choice = input("\nSelect model [1-3] (Enter to keep current): ").strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                self._config.model = options[idx][0]
                self._config.save()
                self._console.print(f"[green]Model set to {self._config.model}[/green]")
                self._build_runner()

    def _retrain(self) -> None:
        from pinpoint.recommender.trainer import train

        self._console.print("[cyan]Retraining recommender...[/cyan]")
        try:
            model = train(data_dir=self._config.data_dir)
            self._console.print(
                f"[green]OK Trained on {model.n_states} states, "
                f"{len(model.feature_columns)} features[/green]"
            )
        except Exception as e:
            self._console.print(f"[red]Training failed:[/red] {e}")

    def _import_data(self, path: str) -> None:
        import shutil
        from pathlib import Path
        from pinpoint.agent.config import CUSTOM_DATA_DIR

        src = Path(path)
        if not src.exists():
            self._console.print(f"[red]File not found:[/red] {path}")
            return

        dest = CUSTOM_DATA_DIR / src.name
        CUSTOM_DATA_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        self._console.print(f"[green]OK Imported {src.name} → {dest}[/green]")
        self._console.print("[dim]Run /train to apply the new data.[/dim]")

    # ------------------------------------------------------------------
    def _toolbar(self) -> HTML:
        model_short = self._config.model.split("-")[1] if "-" in self._config.model else self._config.model
        return HTML(
            f" <b>{model_short}</b>   "
            "<style bg='#1a1a2e' fg='#555577'>/help for commands  ·  Ctrl+C × 2 to exit</style>"
        )

    def _make_bindings(self) -> KeyBindings:
        kb = KeyBindings()

        @kb.add("c-c")
        def _(event):
            raise KeyboardInterrupt

        return kb

"""Pinpoint CLI — entry point and all commands."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from pinpoint.__init__ import __version__
from pinpoint.agent.config import Config, MODELS

app = typer.Typer(
    name="pinpoint",
    help="AI-powered business location recommender.",
    invoke_without_command=True,
    no_args_is_help=False,
    add_completion=False,
    pretty_exceptions_enable=False,
)
# Force UTF-8 output on Windows to avoid GBK codec errors
console = Console(highlight=False, emoji=False)

config_app = typer.Typer(help="Manage configuration.", no_args_is_help=True)
app.add_typer(config_app, name="config")


# --------------------------------------------------------------------------- #
# Default: launch REPL                                                         #
# --------------------------------------------------------------------------- #

@app.callback(invoke_without_command=True)
def _default(ctx: typer.Context) -> None:
    """Launch the interactive REPL when no subcommand is given."""
    if ctx.invoked_subcommand is not None:
        return

    config = Config.load()
    from pinpoint.ui.terminal import InteractiveTerminal
    terminal = InteractiveTerminal(config)
    terminal.run()


# --------------------------------------------------------------------------- #
# setup                                                                        #
# --------------------------------------------------------------------------- #

@app.command()
def setup() -> None:
    """First-time setup wizard — configure API key, model, and train the model."""
    config = Config.load()
    console.print(f"\n[bold #00d4ff]Pinpoint Setup[/bold #00d4ff]  v{__version__}\n")

    # Step 1: API key
    console.rule("[dim]Step 1 / 3 — Anthropic API Key[/dim]")
    current = config.masked_key
    console.print(f"  Current key: [dim]{current}[/dim]")
    new_key = input("  Enter API key (or Enter to keep current): ").strip()
    if new_key:
        config.api_key = new_key
        console.print("  [green]OK Key updated[/green]")

    # Step 2: Model
    console.rule("[dim]Step 2 / 3 — Default Model[/dim]")
    options = list(MODELS.items())
    for i, (mid, label) in enumerate(options, 1):
        marker = "[bold cyan]*[/bold cyan]" if mid == config.model else " "
        console.print(f"  {marker} [{i}] {label}")

    choice = input(f"\n  Select model [1-{len(options)}] (Enter to keep current): ").strip()
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(options):
            config.model = options[idx][0]
            console.print(f"  [green]OK Model: {config.model}[/green]")

    # Save config
    config.save()
    console.print(f"\n  [dim]Config saved to ~/.pinpoint/config.json[/dim]")

    # Step 3: Train model
    console.rule("[dim]Step 3 / 3 — Train Recommender[/dim]")
    answer = input("  Train the location recommender now? [Y/n]: ").strip().lower()
    if answer in ("", "y", "yes"):
        from pinpoint.recommender.trainer import train
        console.print("  [cyan]Training…[/cyan]")
        try:
            model = train(quiet=False)
            console.print(
                f"  [green]OK Trained on {model.n_states} states, "
                f"{len(model.feature_columns)} features[/green]"
            )
        except Exception as e:
            console.print(f"  [red]Training failed:[/red] {e}")
            console.print("  [dim]You can train later with: pinpoint train[/dim]")

    # Done
    console.print()
    console.print(
        Panel(
            "[green]Setup complete![/green]\n\n"
            "  Launch REPL:          [cyan]pinpoint[/cyan]\n"
            "  One-shot recommend:   [cyan]pinpoint recommend --priorities talent,gdp_growth[/cyan]\n"
            "  Check status:         [cyan]pinpoint doctor[/cyan]",
            border_style="green",
        )
    )


# --------------------------------------------------------------------------- #
# recommend                                                                    #
# --------------------------------------------------------------------------- #

@app.command()
def recommend(
    priorities: str = typer.Option(
        ..., "--priorities", "-p",
        help=(
            "Comma-separated priorities: talent, cost_efficiency, gdp_growth, "
            "manufacturing, tech_ecosystem, finance, healthcare, logistics, energy"
        ),
    ),
    describe: Optional[str] = typer.Option(
        None, "--describe", "-d",
        help="Natural language business description (uses Claude to parse if API key set)",
    ),
    top: int = typer.Option(5, "--top", "-n", help="Number of top results"),
    region: Optional[str] = typer.Option(None, "--region", "-r", help="Preferred US region"),
) -> None:
    """Recommend US states for your business (one-shot, no REPL)."""
    config = Config.load()

    priority_list = [p.strip() for p in priorities.split(",") if p.strip()]

    from pinpoint.recommender.trainer import load_or_train
    console.print("[cyan]Loading recommender…[/cyan]")
    try:
        model = load_or_train(data_dir=config.data_dir)
    except Exception as e:
        console.print(f"[red]Error loading model:[/red] {e}")
        console.print("[dim]Run `pinpoint train` first.[/dim]")
        raise typer.Exit(1)

    console.print(f"[dim]Priorities: {', '.join(priority_list)}[/dim]\n")
    results = model.recommend(priority_list, top_n=top, preferred_region=region)

    table = Table(
        title="Top Location Recommendations",
        box=box.ROUNDED,
        show_lines=False,
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("State", style="bold")
    table.add_column("Score", style="cyan", justify="right")
    table.add_column("Key Strengths", style="dim")

    for r in results:
        strengths = ", ".join(f[0].replace("_", " ") for f in r.top_features[:2])
        table.add_row(str(r.rank), r.state_name, f"{r.score:.3f}", strengths or "—")

    console.print(table)


# --------------------------------------------------------------------------- #
# train                                                                        #
# --------------------------------------------------------------------------- #

@app.command()
def train(
    data_dir: Optional[Path] = typer.Option(
        None, "--data-dir", help="Path to directory containing CAGDP9 CSV files"
    ),
) -> None:
    """Train (or retrain) the location recommender on CSV data."""
    from pinpoint.recommender.trainer import train as _train

    config = Config.load()
    resolved_dir = data_dir or config.data_dir

    console.print("[cyan]Training recommender…[/cyan]")
    try:
        model = _train(data_dir=resolved_dir, quiet=False)
        console.print(
            f"\n[green]OK Done.[/green] "
            f"{model.n_states} states | {len(model.feature_columns)} features"
        )
    except Exception as e:
        console.print(f"[red]Training failed:[/red] {e}")
        raise typer.Exit(1)


# --------------------------------------------------------------------------- #
# import-data                                                                  #
# --------------------------------------------------------------------------- #

@app.command(name="import-data")
def import_data(
    path: Path = typer.Argument(..., help="Path to custom CSV file to import"),
) -> None:
    """Import a custom CSV dataset to augment the recommender training data.

    The CSV must have a 'state' column (2-letter code) and numeric feature columns.
    Example columns: state, talent_score, cost_index, infrastructure_score
    """
    import shutil
    from pinpoint.agent.config import CUSTOM_DATA_DIR

    if not path.exists():
        console.print(f"[red]File not found:[/red] {path}")
        raise typer.Exit(1)

    CUSTOM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest = CUSTOM_DATA_DIR / path.name
    shutil.copy2(path, dest)
    console.print(f"[green]OK Imported:[/green] {path.name} → {dest}")
    console.print("[dim]Run `pinpoint train` to apply the new data.[/dim]")


# --------------------------------------------------------------------------- #
# doctor                                                                       #
# --------------------------------------------------------------------------- #

@app.command()
def doctor() -> None:
    """Check system status — API key, model, recommender, data."""
    config = Config.load()
    console.print(f"\n[bold #00d4ff]Pinpoint Doctor[/bold #00d4ff]  v{__version__}\n")

    # --- Configuration ---
    t = Table(title="Configuration", box=box.SIMPLE_HEAVY, show_header=False)
    t.add_column("key", style="cyan")
    t.add_column("value")
    t.add_row("api_key", config.masked_key)
    t.add_row("model", config.model)
    t.add_row("top_n", str(config.top_n))
    t.add_row("data_dir", str(config.data_dir or "(bundled)"))
    console.print(t)

    # --- API connectivity ---
    t2 = Table(title="API Check", box=box.SIMPLE_HEAVY, show_header=False)
    t2.add_column("check", style="cyan")
    t2.add_column("result")

    if config.api_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=config.api_key)
            client.models.list(limit=1)
            t2.add_row("Anthropic API", "[green]OK Connected[/green]")
        except Exception as e:
            t2.add_row("Anthropic API", f"[red]FAIL {e}[/red]")
    else:
        t2.add_row("Anthropic API", "[yellow]WARN No key set[/yellow]")
    console.print(t2)

    # --- Recommender ---
    t3 = Table(title="Recommender", box=box.SIMPLE_HEAVY, show_header=False)
    t3.add_column("check", style="cyan")
    t3.add_column("result")

    from pinpoint.agent.config import MODEL_FILE
    from pinpoint.recommender.trainer import load

    model = load()
    if model and model.is_fitted:
        t3.add_row("Model file", "[green]OK Trained[/green]")
        t3.add_row("States", str(model.n_states))
        t3.add_row("Features", str(len(model.feature_columns)))
    else:
        t3.add_row("Model file", "[yellow]WARN Not trained — run `pinpoint train`[/yellow]")
    console.print(t3)

    # --- Data ---
    t4 = Table(title="Data", box=box.SIMPLE_HEAVY, show_header=False)
    t4.add_column("check", style="cyan")
    t4.add_column("result")

    from pinpoint.recommender.trainer import _default_data_dir
    try:
        data_dir = config.data_dir or _default_data_dir()
        csv_count = len(list(data_dir.glob("CAGDP9_*.csv")))
        t4.add_row("Bundled CSVs", f"[green]OK {csv_count} files[/green]  [dim]{data_dir}[/dim]")
    except FileNotFoundError as e:
        t4.add_row("Bundled CSVs", f"[red]FAIL {e}[/red]")

    from pinpoint.agent.config import CUSTOM_DATA_DIR
    custom_files = list(CUSTOM_DATA_DIR.glob("*.csv")) if CUSTOM_DATA_DIR.exists() else []
    t4.add_row(
        "Custom CSVs",
        f"[green]{len(custom_files)} file(s)[/green]" if custom_files else "[dim]none[/dim]",
    )
    console.print(t4)


# --------------------------------------------------------------------------- #
# config subcommands                                                           #
# --------------------------------------------------------------------------- #

@config_app.command("show")
def config_show() -> None:
    """Show current configuration."""
    config = Config.load()
    t = Table(box=box.SIMPLE_HEAVY, show_header=False)
    t.add_column("key", style="cyan")
    t.add_column("value")
    for k, v in config.as_dict().items():
        display = config.masked_key if k == "api_key" else str(v)
        t.add_row(k, display)
    console.print(t)


@config_app.command("set")
def config_set(
    key: str = typer.Argument(..., help="Config key"),
    value: str = typer.Argument(..., help="Config value"),
) -> None:
    """Set a configuration value."""
    config = Config.load()
    valid_keys = {"api_key", "model", "top_n", "data_dir"}
    if key not in valid_keys:
        console.print(f"[red]Unknown key:[/red] {key}")
        console.print(f"[dim]Valid keys: {', '.join(sorted(valid_keys))}[/dim]")
        raise typer.Exit(1)

    if key == "top_n":
        setattr(config, key, int(value))
    else:
        setattr(config, key, value)
    config.save()
    console.print(f"[green]OK[/green] {key} = {value}")


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #

def entry() -> None:
    app()


if __name__ == "__main__":
    entry()

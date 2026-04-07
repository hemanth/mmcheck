"""CLI for mmcheck."""

import argparse
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from mmcheck import __version__
from mmcheck.core import check


def _render(info, console: Console) -> None:
    """Render ModelInfo as a rich panel."""
    mm_badge = Text("YES", style="bold green") if info.multimodal else Text("NO", style="dim")

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="bold cyan", min_width=12)
    table.add_column("Value")

    table.add_row("Model", info.name)
    table.add_row("Multimodal", mm_badge)
    table.add_row("Inputs", Text(info.input_str, style="green" if info.multimodal else ""))
    table.add_row("Outputs", Text(info.output_str))

    if info.architecture:
        table.add_row("Architecture", info.architecture)
    if info.model_type:
        table.add_row("Type", info.model_type)

    table.add_row("Source", Text(info.source, style="dim"))

    title = "mmcheck"
    border = "green" if info.multimodal else "dim"
    console.print(Panel(table, title=title, border_style=border, padding=(1, 2)))


def main():
    parser = argparse.ArgumentParser(
        prog="mmcheck",
        description="Check if a model supports multimodal inputs",
    )
    parser.add_argument(
        "model",
        nargs="?",
        help="Model name or HuggingFace model ID",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip HuggingFace Hub lookup (registry only)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Output as JSON",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"mmcheck {__version__}",
    )

    args = parser.parse_args()
    console = Console()

    if not args.model:
        console.print("[bold]mmcheck[/bold] - Check model multimodal capabilities")
        console.print("Usage: mmcheck <model-name>")
        console.print("Example: mmcheck google/gemma-3-27b-it")
        sys.exit(0)

    info = check(args.model, offline=args.offline)

    if args.as_json:
        import json
        data = {
            "name": info.name,
            "multimodal": info.multimodal,
            "input_modalities": info.input_modalities,
            "output_modalities": info.output_modalities,
            "architecture": info.architecture,
            "model_type": info.model_type,
            "source": info.source,
        }
        console.print_json(json.dumps(data))
    else:
        _render(info, console)


if __name__ == "__main__":
    main()

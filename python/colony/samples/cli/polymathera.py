import os
import sys
import time
import typer
from rich.logging import RichHandler
import logging
from rich.console import Console
from rich.markdown import Markdown
from rich.columns import Columns
from rich.progress import track
from rich.progress import Progress

from ..utils import setup_logger


logger = setup_logger(__name__)
console = Console()

MARKDOWN = """
# This is an h1

Rich can do a pretty *decent* job of rendering markdown.

1. This is a list item
2. This is another list item
"""



app = typer.Typer()

@app.command()
def summary(entity: str):
    """
    Generate a summary for the specified entity.
    """
    # TODO: Add logic to generate summary
    console.print(f"[green underline]Generating summary for {entity}[/green underline]")
    with console.status("Stage 1: Working..."):
        time.sleep(2)
    with console.status("Stage 2: Working...", spinner="monkey"):
        time.sleep(2)

    text = f"Generated summary for [i]{entity}[/i] :smiley:"
    console.print(text, style="bold white on blue", justify="left")
    console.print(text, style="bold white on blue", justify="center")
    console.print(text, style="bold white on blue", justify="right")
    console1 = Console(width=14)
    console.rule("ellipsis")
    console1.print(text, style="bold white on blue", justify="left", overflow="ellipsis")

    console.rule("Markdown")
    md = Markdown(MARKDOWN)
    console.print(md)

    console.rule("Columns")
    directory = os.listdir('.')
    columns = Columns(directory, equal=True, expand=True)
    console.print(columns)

    # TODO: Progress has so many options, need to explore more
    console.rule("Progress")
    with Progress(transient=True) as progress:

        task1 = progress.add_task("[red]Downloading...", total=1000)
        task2 = progress.add_task("[green]Processing...", total=1000)
        task3 = progress.add_task("[cyan]Cooking...", total=1000)

        while not progress.finished:
            progress.update(task1, advance=0.7)
            progress.update(task2, advance=0.5)
            progress.update(task3, advance=0.9)
            time.sleep(0.02)

@app.command()
def deploy(
    environment: str = typer.Option(..., help="Deployment environment"),
    version: str = typer.Option(..., help="Version to deploy"),
):
    """
    Deploy the Polymathera system.
    """
    logger.info(f"Deploying version {version} to {environment}")
    # Add deployment logic

    from rich.__main__ import make_test_card

    with console.pager():
        console.print(make_test_card())



if __name__ == "__main__":
    logger.info("Starting Polymathera CLI")
    app()


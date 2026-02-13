"""colony-env: CLI for managing Colony local test environments.

Usage:
    colony-env up [--workers N] [--no-build] [--k8s]
    colony-env down [--k8s]
    colony-env status
    colony-env run PATH [--config YAML]
    colony-env doctor
"""

from __future__ import annotations

import asyncio
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .config import DeployConfig
from .manager import DeploymentManager
from .providers.base import ProviderStatus

app = typer.Typer(
    name="colony-env",
    help="Manage Colony local test environments.",
    no_args_is_help=True,
)
console = Console()


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.run(coro)


@app.command()
def up(
    workers: int = typer.Option(1, "--workers", "-w", help="Number of Ray workers"),
    no_build: bool = typer.Option(False, "--no-build", help="Skip image build"),
    k8s: bool = typer.Option(False, "--k8s", help="Use Kind + KubeRay (advanced)"),
):
    """Build Colony image and start Ray cluster + Redis."""
    config = DeployConfig(mode="k8s" if k8s else "compose")
    manager = DeploymentManager(config)

    try:
        services = _run(manager.up(
            build=not no_build,
            workers=workers,
            on_status=lambda msg: console.print(f"  [blue]{msg}[/blue]"),
        ))
    except RuntimeError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    all_ok = all(s.status == ProviderStatus.RUNNING for s in services)

    for svc in services:
        icon = "[green]OK[/green]" if svc.status == ProviderStatus.RUNNING else "[red]FAIL[/red]"
        detail = ""
        if svc.details.get("dashboard"):
            detail = f"  dashboard: {svc.details['dashboard']}"
        elif svc.port:
            detail = f"  localhost:{svc.port}"
        if svc.details.get("replicas"):
            detail += f"  (x{svc.details['replicas']})"
        console.print(f"  {svc.name:16s} {icon}{detail}")

    if all_ok:
        console.print()
        console.print("[green]Ready![/green] Run your analysis:")
        console.print("  colony-env run /path/to/codebase --config analysis.yaml")
    else:
        console.print()
        console.print("[red]Some services failed to start. Check 'docker compose logs'.[/red]")
        raise typer.Exit(1)


@app.command()
def down(
    k8s: bool = typer.Option(False, "--k8s", help="Use Kind + KubeRay (advanced)"),
):
    """Stop and remove all containers and resources."""
    config = DeployConfig(mode="k8s" if k8s else "compose")
    manager = DeploymentManager(config)

    try:
        with console.status("[blue]Stopping colony environment..."):
            _run(manager.down())
    except RuntimeError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    console.print("[green]Colony environment stopped.[/green]")


@app.command()
def status():
    """Show status of all running services."""
    manager = DeploymentManager()
    services = _run(manager.status())

    if not services:
        console.print("No colony services running.")
        return

    table = Table(title="Colony Environment")
    table.add_column("Service", style="bold")
    table.add_column("Status")
    table.add_column("Details")

    for svc in services:
        if svc.status == ProviderStatus.RUNNING:
            status_str = "[green]running[/green]"
        elif svc.status == ProviderStatus.STOPPED:
            status_str = "[dim]stopped[/dim]"
        else:
            status_str = f"[red]{svc.status.value}[/red]"

        details = ", ".join(f"{k}={v}" for k, v in svc.details.items())
        table.add_row(svc.name, status_str, details)

    console.print(table)


@app.command()
def run(
    codebase_path: str = typer.Argument(..., help="Path to the codebase to analyze"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to analysis YAML config"),
    k8s: bool = typer.Option(False, "--k8s", help="Use Kind + KubeRay (advanced)"),
):
    """Run polymath.py analysis inside the cluster."""
    deploy_config = DeployConfig(mode="k8s" if k8s else "compose")
    manager = DeploymentManager(deploy_config)

    try:
        exit_code = _run(manager.run(
            codebase_path=codebase_path,
            config_path=config,
        ))
    except (RuntimeError, FileNotFoundError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    raise typer.Exit(exit_code)


@app.command()
def doctor():
    """Check prerequisites for running colony-env."""
    manager = DeploymentManager()
    checks = _run(manager.doctor())

    all_ok = True
    for name, passed in checks.items():
        icon = "[green]OK[/green]" if passed else "[red]MISSING[/red]"
        console.print(f"  {name:20s} {icon}")
        if not passed:
            all_ok = False

    if all_ok:
        console.print()
        console.print("[green]All prerequisites met.[/green]")
    else:
        console.print()
        console.print("[red]Some prerequisites are missing. Install Docker and try again.[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()

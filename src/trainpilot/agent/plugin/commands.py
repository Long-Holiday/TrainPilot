"""``plugin ...`` -- install the GPU-side integration into Antigravity CLI.

The GPU node needs no TrainPilot code to *use* the control plane: Antigravity has its own
remote MCP client. What it cannot know is Antigravity's spelling of "here is a remote MCP
server", which is where an install usually fails -- quietly, with an agent that starts up
fine and never sees the tools. So ``install`` renders the plugin in Antigravity's own
layout, writes it to a staging directory, and then hands that directory to ``agy plugin
install`` rather than guessing where Antigravity keeps its plugins.

Nothing here needs the control plane's own dependencies, a running server, or an installed
TrainPilot package: this module is standard library only, and ``scripts/install_agent.py``
drives it with a bare ``python3``.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from trainpilot.agent.plugin import catalog
from trainpilot.agent.plugin import install as installer
from trainpilot.agent.plugin import render, templates

STAGING_HOME = "~/.trainpilot/plugins"

EXIT_OK = 0
EXIT_FAILED = 1


# --------------------------------------------------------------------------- shared


def _add_connection_flags(parser: argparse.ArgumentParser) -> None:
    """Let the connection flags be written at any level of the command tree.

    ``SUPPRESS`` as the default means a leaf parser leaves the inherited value alone when
    the flag is not repeated there, so ``--server-url`` before or after the subcommand
    behaves the same.
    """
    parser.add_argument("--server-url", "-s", default=argparse.SUPPRESS)
    parser.add_argument("--api-token", default=argparse.SUPPRESS)
    parser.add_argument("--task-id", "-t", default=argparse.SUPPRESS)


def add_parser(subparsers: argparse._SubParsersAction, prog: Optional[str] = None) -> None:
    """Register the ``plugin`` command group on a parser.

    ``prog`` overrides the displayed program name, for the standalone installer script.
    """
    parser = subparsers.add_parser(
        "plugin",
        prog=prog,
        help="Install TrainPilot into Antigravity CLI",
        description=(
            "Render TrainPilot into Antigravity's native plugin format and install it with "
            "the agy CLI."
        ),
    )
    _add_connection_flags(parser)

    sub = parser.add_subparsers(dest="plugin_command", required=True, help="Plugin subcommands")

    p_list = sub.add_parser("list", help="Show the target agent and what this node already has")
    p_list.set_defaults(func=cmd_list)

    for name, help_text in (
        ("install", "Render and install the plugin into Antigravity CLI"),
        ("uninstall", "Remove the plugin again, leaving nothing else behind"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument(
            "--dry-run",
            action="store_true",
            help="Print what would change and write nothing",
        )
        _add_connection_flags(p)
        p.set_defaults(func=cmd_install if name == "install" else cmd_uninstall)

    p_render = sub.add_parser(
        "render",
        help="Write the plugin out without installing it",
        description=(
            "Materialise the plugin bundle for inspection, or to hand to "
            "`agy plugin install` yourself."
        ),
    )
    p_render.add_argument("--out", help="Directory to write the bundle into")
    _add_connection_flags(p_render)
    p_render.set_defaults(func=cmd_render)


def _context(args) -> render.PluginContext:
    """Build a context from the flags, falling back to the same env the agent reads."""
    from trainpilot.common.gateway import resolve_gateway_url

    return render.PluginContext(
        gateway_url=resolve_gateway_url(getattr(args, "server_url", None)),
        api_token=getattr(args, "api_token", None) or os.getenv("TRAINPILOT_API_TOKEN"),
        task_id=getattr(args, "task_id", None) or os.getenv("TRAINPILOT_TASK_ID", "train-task-default"),
    )


def _staging_dir() -> Path:
    """Where the rendered plugin is staged before ``agy plugin install`` copies it."""
    return installer.expand(f"{STAGING_HOME}/{templates.PLUGIN_NAME}")


# --------------------------------------------------------------------------- commands


def cmd_list(args) -> int:
    """Show the target agent and what this node already has."""
    ctx = _context(args)
    target = catalog.ANTIGRAVITY
    staging = _staging_dir()

    print(f"Gateway   {ctx.gateway_url}")
    print(f"MCP       {ctx.mcp_url}")
    print(f"Task id   {ctx.task_id}")
    print(f"Token     {'set' if ctx.api_token else 'not set (open control plane)'}")
    print()
    print(f"{'AGENT':<14}{'NAME':<20}{'REMOTE SERVER KEY':<22}STATUS")
    status = f"{target.agent_cli} found" if shutil.which(target.agent_cli) else "agy not on PATH"
    print(f"{target.key:<14}{target.display_name:<20}{target.transport:<22}{status}")
    print()
    print(f"Staged    {'yes' if staging.is_dir() else 'no'}  {installer.display_path(staging)}")
    print("Installed plugins are listed by `agy plugin list`.")
    return EXIT_OK


def cmd_install(args) -> int:
    """Render, write, and register the plugin."""
    ctx = _context(args)
    bundle = render.render_bundle(ctx)
    staging = _staging_dir()
    target = catalog.ANTIGRAVITY

    print(f"=== {target.display_name} ===")
    print(f"    {target.docs_url}")

    changes = installer.plan(bundle, staging)
    for line in installer.apply(changes, dry_run=args.dry_run):
        print(f"    {line}")

    if args.dry_run:
        print("    (dry run: nothing was written)")
        return EXIT_OK

    ok = _register(bundle, staging)
    for note in bundle.notes:
        print(f"    note: {note}")
    return EXIT_OK if ok else EXIT_FAILED


def _register(bundle: render.Bundle, staging: Path) -> bool:
    """Hand the staged bundle to ``agy``, or tell the user how to."""
    argv = installer.expand_argv(bundle.register[0], bundle=staging, mcp_url=bundle.mcp_url)
    hint = " ".join(argv)
    if not shutil.which(catalog.ANTIGRAVITY.agent_cli):
        print("    agy is not on PATH, so the plugin was staged but not registered.")
        print(f"    Run this once it is installed: {hint}")
        return False

    print(f"    $ {hint}")
    result = subprocess.run(argv, check=False)
    if result.returncode != 0:
        print(f"    registration failed with exit code {result.returncode}.", file=sys.stderr)
        print(f"    The files are staged at {staging}; retry with: {hint}")
        return False
    return True


def cmd_uninstall(args) -> int:
    """Take the plugin back out of Antigravity and drop the staging directory."""
    target = catalog.ANTIGRAVITY
    staging = _staging_dir()
    print(f"=== {target.display_name} ===")

    if target.unregister and shutil.which(target.agent_cli):
        argv = [
            part.replace("{bundle}", str(staging)).replace("{mcp_url}", "")
            for part in target.unregister[0]
        ]
        print(f"    $ {' '.join(argv)}")
        if not args.dry_run:
            subprocess.run(argv, check=False)
    else:
        print("    agy is not on PATH; skipping its unregister step.")

    if staging.is_dir():
        print(f"    {'delete':<9}  {installer.display_path(staging)}")
        if not args.dry_run:
            shutil.rmtree(staging)

    if args.dry_run:
        print("    (dry run: nothing was removed)")
    return EXIT_OK


def cmd_render(args) -> int:
    """Write the plugin out without touching Antigravity's configuration."""
    ctx = _context(args)
    bundle = render.render_bundle(ctx)
    out = installer.expand(args.out) if args.out else _staging_dir()

    for item in bundle.files:
        destination = out / item.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(item.content, encoding="utf-8")
        print(f"wrote  {destination}")

    argv = installer.expand_argv(bundle.register[0], bundle=out, mcp_url=bundle.mcp_url)
    print(f"\nregister with:\n  {' '.join(argv)}")
    return EXIT_OK

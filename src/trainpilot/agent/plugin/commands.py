"""``plugin ...`` -- install the GPU-side integration into a supported agent.

Each agent has its own remote MCP client and plugin format. ``install`` renders the
selected native format into TrainPilot's staging directory and registers it through
the agent's CLI.

Nothing here needs the control plane's own dependencies, a running server, or an installed
TrainPilot package: this module is standard library only, and ``scripts/install_agent.py``
drives it with a bare ``python3``.
"""

import argparse
import os
import shlex
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
    parser.add_argument(
        "--agent", choices=tuple(catalog.TARGETS), default=argparse.SUPPRESS,
        help="Target agent (default: antigravity)",
    )
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
        help="Install TrainPilot into Antigravity, Claude Code or Codex",
        description="Render a native plugin and register it with the selected agent.",
    )
    _add_connection_flags(parser)

    sub = parser.add_subparsers(dest="plugin_command", required=True, help="Plugin subcommands")

    p_list = sub.add_parser("list", help="Show supported agents and staged plugins")
    _add_connection_flags(p_list)
    p_list.set_defaults(func=cmd_list)

    for name, help_text in (
        ("install", "Render and install the plugin into the selected agent"),
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
        description="Materialise the native plugin bundle for inspection or manual installation.",
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


def _target(args) -> catalog.AgentTarget:
    return catalog.TARGETS[getattr(args, "agent", "antigravity")]


def _staging_dir(target: catalog.AgentTarget = catalog.ANTIGRAVITY) -> Path:
    """Keep the existing Antigravity path and isolate each additional agent's files."""
    suffix = "" if target == catalog.ANTIGRAVITY else f"/{target.key}"
    return installer.expand(f"{STAGING_HOME}{suffix}/{templates.PLUGIN_NAME}")


# --------------------------------------------------------------------------- commands


def cmd_list(args) -> int:
    """Show the supported agents and their independent staging directories."""
    ctx = _context(args)
    targets = [_target(args)] if hasattr(args, "agent") else catalog.TARGETS.values()

    print(f"Gateway   {ctx.gateway_url}")
    print(f"MCP       {ctx.mcp_url}")
    print(f"Task id   {ctx.task_id}")
    print(f"Token     {'set' if ctx.api_token else 'not set (open control plane)'}")
    for target in targets:
        staging = _staging_dir(target)
        status = "found" if shutil.which(target.agent_cli) else "not on PATH"
        print(f"\n{target.key}  {target.display_name}  {target.transport}")
        print(f"CLI       {target.agent_cli} {status}")
        print(f"Staged    {'yes' if staging.is_dir() else 'no'}  {installer.display_path(staging)}")
    print("\nUse the selected agent's plugin manager to inspect installed plugins.")
    return EXIT_OK


def cmd_install(args) -> int:
    """Render, write, and register the plugin."""
    ctx = _context(args)
    target = _target(args)
    bundle = render.render_bundle(ctx, target)
    staging = _staging_dir(target)

    print(f"=== {target.display_name} ===")
    print(f"    {target.docs_url}")

    changes = installer.plan(bundle, staging)
    for line in installer.apply(changes, dry_run=args.dry_run):
        print(f"    {line}")

    if args.dry_run:
        for template in bundle.register:
            argv = installer.expand_argv(template, bundle=staging, mcp_url=bundle.mcp_url)
            print(f"    $ {shlex.join(argv)}")
        print("    (dry run: nothing was written)")
        return EXIT_OK

    ok = _register(bundle, staging, target)
    for note in bundle.notes:
        print(f"    note: {note}")
    return EXIT_OK if ok else EXIT_FAILED


def _register(bundle: render.Bundle, staging: Path, target: catalog.AgentTarget) -> bool:
    """Run each native registration step, stopping if any step fails."""
    if not shutil.which(target.agent_cli):
        print(f"    {target.agent_cli} is not on PATH, so the plugin was staged but not registered.")
        for template in bundle.register:
            argv = installer.expand_argv(template, bundle=staging, mcp_url=bundle.mcp_url)
            print(f"    Run this once it is installed: {shlex.join(argv)}")
        return False

    for template in bundle.register:
        argv = installer.expand_argv(template, bundle=staging, mcp_url=bundle.mcp_url)
        hint = shlex.join(argv)
        print(f"    $ {hint}")
        result = subprocess.run(argv, check=False)
        if result.returncode != 0:
            print(f"    registration failed with exit code {result.returncode}.", file=sys.stderr)
            print(f"    The files are staged at {staging}; retry with: {hint}")
            return False
    return True


def cmd_uninstall(args) -> int:
    """Unregister the selected plugin before removing its staging directory."""
    target = _target(args)
    staging = _staging_dir(target)
    print(f"=== {target.display_name} ===")

    if target.unregister and shutil.which(target.agent_cli):
        for template in target.unregister:
            argv = installer.expand_argv(template, bundle=staging, mcp_url="")
            print(f"    $ {shlex.join(argv)}")
            if not args.dry_run:
                result = subprocess.run(argv, check=False)
                if result.returncode != 0:
                    print("    Unregister failed; staged files were kept for retry.", file=sys.stderr)
                    return EXIT_FAILED
    else:
        print(f"    {target.agent_cli} is not on PATH; skipping its unregister step.")

    if staging.is_dir():
        print(f"    {'delete':<9}  {installer.display_path(staging)}")
        if not args.dry_run:
            shutil.rmtree(staging)

    if args.dry_run:
        print("    (dry run: nothing was removed)")
    return EXIT_OK


def cmd_render(args) -> int:
    """Write the native bundle without changing the agent's configuration."""
    ctx = _context(args)
    target = _target(args)
    bundle = render.render_bundle(ctx, target)
    out = installer.expand(args.out) if args.out else _staging_dir(target)

    for line in installer.apply(installer.plan(bundle, out)):
        print(line)

    print("\nregister with:")
    for template in bundle.register:
        argv = installer.expand_argv(template, bundle=out, mcp_url=bundle.mcp_url)
        print(f"  {shlex.join(argv)}")
    return EXIT_OK

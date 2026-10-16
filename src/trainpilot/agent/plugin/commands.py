"""``plugin ...`` -- install the GPU-side integration into a coding agent.

The GPU node needs no TrainPilot code to *use* the control plane: every supported agent has
its own remote MCP client. What it cannot know is each vendor's spelling of "here is a remote
MCP server", which is where an install usually fails -- quietly, with an agent that starts up
fine and never sees the tools. So ``install`` renders the right layout for the agent in front
of it, writes it, and then hands the result to that agent's own CLI rather than guessing where
the agent keeps its plugins.

Nothing here needs the control plane's own dependencies, a running server, or an installed
TrainPilot package: this module is standard library only, and ``scripts/install_agent.py``
drives it with a bare ``python3``.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from trainpilot.agent.plugin import catalog
from trainpilot.agent.plugin import install as installer
from trainpilot.agent.plugin import render

STAGING_HOME = "~/.trainpilot/plugins"
STAGING_PROJECT = ".trainpilot"

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
        help="Install TrainPilot into a GPU-side coding agent",
        description=(
            "Render TrainPilot into a coding agent's native plugin format and install it. "
            f"Supported agents: {', '.join(catalog.target_keys())}."
        ),
    )
    _add_connection_flags(parser)

    sub = parser.add_subparsers(dest="plugin_command", required=True, help="Plugin subcommands")

    p_list = sub.add_parser("list", help="Show the supported agents and what is installed")
    p_list.set_defaults(func=cmd_list)

    for name, help_text in (
        ("install", "Render and install the plugin for one agent, or every agent found here"),
        ("uninstall", "Remove the plugin again, leaving everything else in the file intact"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument(
            "agent",
            help=f"Agent key, or 'all'. One of: {', '.join(catalog.target_keys())}",
        )
        p.add_argument(
            "--scope",
            choices=catalog.SCOPES,
            default=catalog.USER,
            help="user (default): this machine. project: the current directory only",
        )
        p.add_argument(
            "--dry-run",
            action="store_true",
            help="Print what would change and write nothing",
        )
        _add_connection_flags(p)
        p.set_defaults(func=cmd_install if name == "install" else cmd_uninstall)

    p_render = sub.add_parser(
        "render",
        help="Print or materialise the plugin without installing it",
        description=(
            "For agents with a plugin package this writes the bundle under --out. For agents "
            "installed in place it prints the files that would be edited and their content, "
            "since there is no bundle to inspect."
        ),
    )
    p_render.add_argument("agent", help=f"Agent key. One of: {', '.join(catalog.target_keys())}")
    p_render.add_argument("--out", help="Directory to write the bundle into")
    p_render.add_argument("--scope", choices=catalog.SCOPES, default=catalog.USER)
    _add_connection_flags(p_render)
    p_render.set_defaults(func=cmd_render)


def _context(args) -> render.PluginContext:
    """Build a context from the flags, falling back to the same env the agent reads."""
    from trainpilot.common.gateway import resolve_gateway_url

    return render.PluginContext(
        gateway_url=resolve_gateway_url(getattr(args, "server_url", None)),
        api_token=getattr(args, "api_token", None) or os.getenv("TRAINPILOT_API_TOKEN"),
        task_id=getattr(args, "task_id", None) or os.getenv("TRAINPILOT_TASK_ID", "train-task-default"),
        scope=getattr(args, "scope", catalog.USER),
    )


def _staging_dir(target: str, scope: str) -> Path:
    if scope == catalog.PROJECT:
        return installer.expand(f"{STAGING_PROJECT}/{target}")
    return installer.expand(f"{STAGING_HOME}/{target}")


def _is_detected(target: catalog.AgentTarget) -> bool:
    """Whether this node looks like it has the agent: its CLI, or its config directory."""
    if target.agent_cli and shutil.which(target.agent_cli):
        return True
    return any(installer.expand(path).exists() for path in target.detect)


def _resolve_targets(name: str) -> Optional[List[str]]:
    """Expand an ``all`` argument, or validate one name. ``None`` means it was invalid."""
    if name != "all":
        try:
            catalog.get_target(name)
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return None
        return [name]

    found = [
        key
        for key, target in catalog.TARGETS.items()
        if key != "generic" and _is_detected(target)
    ]
    if not found:
        print(
            "No supported agent was detected on this node. Name one explicitly, or use "
            f"'generic' for the copy-paste config. Supported: {', '.join(catalog.target_keys())}",
            file=sys.stderr,
        )
        return None
    return found


# --------------------------------------------------------------------------- commands


def cmd_list(args) -> int:
    """Show every supported agent and what this node already has."""
    ctx = _context(args)
    print(f"Gateway   {ctx.gateway_url}")
    print(f"MCP       {ctx.mcp_url}")
    print(f"Task id   {ctx.task_id}")
    print(f"Token     {'set' if ctx.api_token else 'not set (open control plane)'}")
    print()
    print(f"{'AGENT':<14}{'NAME':<32}{'REMOTE SERVER KEY':<36}STATUS")
    for key, target in catalog.TARGETS.items():
        if key == "generic":
            status = "always available"
        elif _is_detected(target):
            status = "detected"
        else:
            status = "not found here"
        print(f"{key:<14}{target.display_name:<32}{target.transport:<36}{status}")
    return EXIT_OK


def cmd_install(args) -> int:
    """Render, write, and register the plugin for one agent or every detected one."""
    targets = _resolve_targets(args.agent)
    if targets is None:
        return EXIT_FAILED

    ctx = _context(args)
    failures = 0
    for key in targets:
        failures += _install_one(key, ctx, dry_run=args.dry_run)
    return EXIT_FAILED if failures else EXIT_OK


def _install_one(key: str, ctx: render.PluginContext, *, dry_run: bool) -> int:
    target = catalog.get_target(key)
    bundle = render.render_bundle(key, ctx)
    staging = _staging_dir(key, ctx.scope) if target.kind == catalog.BUNDLE else None

    print(f"\n=== {target.display_name} ({key}) ===")
    print(f"    {target.docs_url}")

    changes = installer.plan(bundle, staging)
    for line in installer.apply(changes, dry_run=dry_run):
        print(f"    {line}")

    if dry_run:
        print("    (dry run: nothing was written)")
        return EXIT_OK

    ok = _register(bundle, staging, ctx)
    for note in bundle.notes:
        print(f"    note: {note}")
    return 0 if ok else EXIT_FAILED


def _register(bundle: render.Bundle, staging: Optional[Path], ctx: render.PluginContext) -> bool:
    """Hand a staged bundle to the agent's own CLI, or tell the user how to."""
    if not bundle.register:
        if bundle.target == "generic":
            entry = {"url": ctx.mcp_url}
            if ctx.headers:
                entry["headers"] = ctx.headers
            print("    Add this entry to your agent's remote MCP configuration:")
            print(json.dumps({"mcpServers": {"trainpilot": entry}}, indent=2))
        return True

    argv_sets = [
        installer.expand_argv(argv, bundle=staging, scope=ctx.scope, mcp_url=bundle.mcp_url)
        for argv in bundle.register
    ]
    target = catalog.get_target(bundle.target)
    if target.agent_cli and not shutil.which(target.agent_cli):
        print(f"    {target.agent_cli} is not on PATH, so the plugin was staged but not registered.")
        print(f"    Run this once it is installed: {_hint(bundle, staging, ctx)}")
        return False

    for argv in argv_sets:
        print(f"    $ {' '.join(argv)}")
        result = subprocess.run(argv, check=False)
        if result.returncode != 0:
            print(f"    registration failed with exit code {result.returncode}.", file=sys.stderr)
            print(f"    The files are staged at {staging}; retry with: {_hint(bundle, staging, ctx)}")
            return False
    return True


def _hint(bundle: render.Bundle, staging: Optional[Path], ctx: render.PluginContext) -> str:
    """The one-line manual fallback for a target whose agent CLI is missing."""
    if bundle.register:
        argv = installer.expand_argv(
            bundle.register[0], bundle=staging, scope=ctx.scope, mcp_url=bundle.mcp_url
        )
        return " ".join(argv)
    return bundle.manual_hint.replace("{mcp_url}", bundle.mcp_url)


def cmd_uninstall(args) -> int:
    """Take the plugin back out of one agent, or every agent installed here."""
    targets = _resolve_targets(args.agent)
    if targets is None:
        return EXIT_FAILED

    ctx = _context(args)
    for key in targets:
        target = catalog.get_target(key)
        bundle = render.render_bundle(key, ctx)
        print(f"\n=== {target.display_name} ({key}) ===")

        staging = _staging_dir(key, ctx.scope) if target.kind == catalog.BUNDLE else None
        if bundle.unregister and not args.dry_run:
            _unregister(bundle, ctx)

        if target.kind == catalog.IN_PLACE:
            for line in installer.apply(installer.plan_revert(bundle), dry_run=args.dry_run):
                print(f"    {line}")

        if staging is not None and staging.exists():
            print(f"    {'delete':<9}  {installer.display_path(staging)}")
            if not args.dry_run:
                shutil.rmtree(staging)

        if args.dry_run:
            print("    (dry run: nothing was removed)")
    return EXIT_OK


def _unregister(bundle: render.Bundle, ctx: render.PluginContext) -> None:
    target = catalog.get_target(bundle.target)
    if target.agent_cli and not shutil.which(target.agent_cli):
        print(f"    {target.agent_cli} is not on PATH; skipping its unregister step.")
        return
    for template in bundle.unregister:
        argv = installer.expand_argv(
            template, bundle=_staging_dir(bundle.target, ctx.scope), scope=ctx.scope, mcp_url=bundle.mcp_url
        )
        print(f"    $ {' '.join(argv)}")
        subprocess.run(argv, check=False)


def cmd_render(args) -> int:
    """Print (or write out) the plugin without touching any agent's configuration."""
    try:
        target = catalog.get_target(args.agent)
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_FAILED

    ctx = _context(args)
    bundle = render.render_bundle(args.agent, ctx)

    if target.kind == catalog.BUNDLE:
        out = installer.expand(args.out) if args.out else _staging_dir(args.agent, ctx.scope)
        for item in bundle.files:
            destination = out / item.path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(item.content, encoding="utf-8")
            print(f"wrote  {destination}")
        if bundle.register:
            argv = installer.expand_argv(
                bundle.register[0], bundle=out, scope=ctx.scope, mcp_url=bundle.mcp_url
            )
            print(f"\nregister with:\n  {' '.join(argv)}")
        return EXIT_OK

    # In-place targets have no bundle to inspect: show what would be edited instead.
    print(f"# {target.display_name} is configured in place. Files that would be written:\n")
    for item in bundle.files:
        print(f"--- {installer.expand(item.path)} ---")
        print(item.content)
    for item in bundle.merges:
        path = installer.expand(item.path)
        print(f"--- {path} ---")
        print(installer._merged_content(path, item))
    print(f"# Manual equivalent:\n#   {_hint(bundle, None, ctx)}")
    return EXIT_OK

"""The dependency split: this package is the control plane, and nothing else.

The GPU side installs no Python at all -- it gets a skill plus a remote MCP entry from a
standard-library installer (``scripts/install_agent.py``) or Claude Code's marketplace, and
its agent's own MCP client talks to ``/mcp``. So every runtime dependency belongs to the
``server`` half, and the base list must stay empty: a GPU node that pip-installs this
package should pull nothing.

The server half is declared twice on purpose -- once as the ``server`` extra for
``pip install trainpilot[server]``, once as the ``server`` dependency group because uv has
no ``default-extras`` -- so the first test here pins the two equal. Without it, a dependency
added to one list silently stops being installed for the other.
"""

import sys
from pathlib import Path

import pytest

if sys.version_info < (3, 11):  # tomllib is stdlib from 3.11
    pytest.skip("pyproject parsing needs tomllib", allow_module_level=True)

import tomllib  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

SERVER_STACK = ("fastapi", "lark-oapi", "mcp", "pydantic", "pydantic-settings", "uvicorn")
TEST_ONLY = ("httpx", "pytest")


def test_server_extra_and_group_are_identical():
    extra = sorted(PYPROJECT["project"]["optional-dependencies"]["server"])
    group = sorted(PYPROJECT["dependency-groups"]["server"])
    assert extra == group, (
        "the `server` extra and the `server` dependency group have drifted apart; "
        "whichever list is missing an entry will install a control plane that cannot boot"
    )


def test_a_gpu_node_pulls_no_dependencies_at_all():
    """Nothing here is needed to install a skill and a remote MCP entry."""
    assert PYPROJECT["project"]["dependencies"] == []


@pytest.mark.parametrize("name", SERVER_STACK)
def test_the_server_stack_lives_in_the_server_extra(name):
    extra = [dep.lower() for dep in PYPROJECT["project"]["optional-dependencies"]["server"]]
    assert any(dep.startswith(name) for dep in extra), (
        f"{name} is part of the control plane and belongs in the `server` extra"
    )


@pytest.mark.parametrize("name", TEST_ONLY)
def test_test_only_tools_are_not_installed_for_the_server(name):
    """pytest and the test client's httpx must not ride along into a deployment image."""
    extra = [dep.lower() for dep in PYPROJECT["project"]["optional-dependencies"]["server"]]
    assert not any(dep.startswith(name) for dep in extra)


def test_the_only_entry_point_is_the_server():
    """A GPU-side CLI would drag the whole Python stack back onto training nodes."""
    assert set(PYPROJECT["project"]["scripts"]) == {"trainpilot-server"}


def test_the_gpu_side_ships_no_python_client():
    """The GPU node runs its agent's MCP client, not ours; a client module here would be
    dead code that the dependency split would have to carry."""
    agent = PROJECT_ROOT / "src" / "trainpilot" / "agent"
    assert sorted(path.name for path in agent.glob("*.py")) == ["__init__.py"]


def test_repo_defaults_install_the_server_group():
    """`uv sync` and `uv run pytest` must keep working without extra flags."""
    assert set(PYPROJECT["tool"]["uv"]["default-groups"]) >= {"dev", "server"}


@pytest.mark.parametrize("name", ["start.sh", "Dockerfile"])
def test_deployment_asks_for_the_server_group(name):
    """The server entry points cannot rely on default-groups alone: an inherited
    UV_NO_DEFAULT_GROUPS would otherwise produce an image that fails at import."""
    text = (PROJECT_ROOT / name).read_text(encoding="utf-8")
    assert "--group server" in text, f"{name} must install the server dependency group"


@pytest.mark.parametrize("name", ["README.md", "README_zh.md"])
def test_docs_only_offer_install_sources_that_exist(name):
    """`uv tool install trainpilot` reads well and cannot work: the project is not on
    PyPI. Every install command the READMEs show has to name a real source."""
    text = (PROJECT_ROOT / name).read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip().lstrip("$ ").strip()
        for prefix in ("uv tool install ", "pip install ", "uvx "):
            if not stripped.startswith(prefix):
                continue
            target = stripped[len(prefix):].strip().strip('"').strip("'")
            assert target.startswith(("git+", "http", ".", "/", "-")), (
                f"{name} tells the reader to install '{target}', which is not published "
                "anywhere; point at the repository or a path"
            )

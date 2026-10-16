"""The dependency split: a GPU node must not inherit the control plane's stack.

``trainpilot.server`` needs FastAPI, uvicorn, lark-oapi and pydantic-settings; the GPU side
needs mcp, httpx and pydantic. The server half is declared twice on purpose -- once as the
``server`` extra for ``pip install trainpilot[server]``, once as the ``server`` dependency
group because uv has no ``default-extras`` -- so the first test here pins the two equal.
Without it, a dependency added to one list silently stops being installed for the other.
"""

import sys
from pathlib import Path

import pytest

if sys.version_info < (3, 11):  # tomllib is stdlib from 3.11
    pytest.skip("pyproject parsing needs tomllib", allow_module_level=True)

import tomllib  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

SERVER_ONLY = ("fastapi", "lark-oapi", "pydantic-settings", "uvicorn")
GPU_ONLY = ("mcp", "httpx", "pydantic")


def test_server_extra_and_group_are_identical():
    extra = sorted(PYPROJECT["project"]["optional-dependencies"]["server"])
    group = sorted(PYPROJECT["dependency-groups"]["server"])
    assert extra == group, (
        "the `server` extra and the `server` dependency group have drifted apart; "
        "whichever list is missing an entry will install a control plane that cannot boot"
    )


@pytest.mark.parametrize("name", SERVER_ONLY)
def test_server_stack_is_not_a_base_dependency(name):
    """A GPU node installing the package must not pull any of these."""
    base = [dep.lower() for dep in PYPROJECT["project"]["dependencies"]]
    assert not any(dep.startswith(name) for dep in base), (
        f"{name} is a control-plane dependency and belongs in the `server` extra"
    )


@pytest.mark.parametrize("name", GPU_ONLY)
def test_gpu_stack_is_a_base_dependency(name):
    base = [dep.lower() for dep in PYPROJECT["project"]["dependencies"]]
    assert any(dep.startswith(name) for dep in base)


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

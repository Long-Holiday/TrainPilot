"""Repo-wide test isolation.

Two things every suite depends on live here so that no test module has to repeat them:

- SQLite is redirected to a temporary database. ``settings.sqlite_path`` defaults to the
  *relative* path ``trainpilot.db``, and the mailbox teardown calls ``DELETE FROM tasks``,
  so an unredirected run would wipe the developer's local task history. The redirect must
  happen before ``trainpilot.server.mailbox`` is imported, because ``default_mailbox``
  builds its storage from settings at import time.
- The shared mailbox and the API/webhook tokens are reset around every test, so a local
  ``.env`` cannot leak credentials into the suite.
"""

import atexit
import os
import shutil
import sys
import tempfile

import pytest

# Clean invalid bracketed IPv6 in NO_PROXY/no_proxy which breaks httpx URLPattern parsing
for var in ("NO_PROXY", "no_proxy"):
    val = os.environ.get(var)
    if val and "[::1]" in val:
        os.environ[var] = val.replace("[::1]", "::1")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (os.path.join(_PROJECT_ROOT, "src"), _PROJECT_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

_TMP_DIR = tempfile.mkdtemp(prefix="trainpilot-tests-")
atexit.register(shutil.rmtree, _TMP_DIR, ignore_errors=True)
TEST_SQLITE_PATH = os.path.join(_TMP_DIR, "trainpilot-test.db")

os.environ["TRAINPILOT_SQLITE_PATH"] = TEST_SQLITE_PATH

from trainpilot.server.config import settings  # noqa: E402  (must follow the redirect)

settings.sqlite_path = TEST_SQLITE_PATH


@pytest.fixture(autouse=True)
def isolate_global_state():
    """Reset the shared mailbox and detach auth from the local .env for every test."""
    from trainpilot.server.mailbox import default_mailbox

    assert default_mailbox._storage.db_path == TEST_SQLITE_PATH, (
        "tests must never touch the repository database"
    )

    default_mailbox.reset()
    saved_tokens = (settings.api_token, settings.feishu_verification_token)
    settings.api_token = None
    settings.feishu_verification_token = None
    yield
    settings.api_token, settings.feishu_verification_token = saved_tokens
    default_mailbox.reset()

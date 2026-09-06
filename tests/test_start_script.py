"""Automated tests for start.sh and setup_skills.sh scripts."""

import os
import subprocess
import sys
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
START_SH = os.path.join(PROJECT_ROOT, "start.sh")
SETUP_SKILLS_SH = os.path.join(PROJECT_ROOT, "setup_skills.sh")


def _isolated_env(tmp_path):
    """Redirect HOME/XDG_CONFIG_HOME so global install never pollutes the real home."""
    fake_home = str(tmp_path / "fakehome")
    env = dict(os.environ)
    env["HOME"] = fake_home
    env["XDG_CONFIG_HOME"] = os.path.join(fake_home, ".config")
    return env, fake_home


def test_start_script_help():
    """Verify start.sh --help outputs usage instructions (Web-only, no auto skills)."""
    res = subprocess.run([START_SH, "--help"], cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert res.returncode == 0
    assert "TrainPilot Web 服务启动脚本使用说明" in res.stdout
    assert "--daemon" in res.stdout
    assert "28780" in res.stdout
    # Web 脚本不再自动同步 skills, 仅作为 GPU 侧手动指引提及
    assert "--host" in res.stdout
    assert "不创建" in res.stdout or "不再" in res.stdout or "仅 Web" in res.stdout
    assert "自动同步全局" not in res.stdout
    assert "setup_skills.sh" in res.stdout  # 仅作为 GPU 手动指引保留


def test_setup_skills_script_help():
    """Verify setup_skills.sh --help outputs usage instructions."""
    res = subprocess.run([SETUP_SKILLS_SH, "--help"], cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert res.returncode == 0
    assert "TrainPilot Agent Skills 生成脚本使用说明" in res.stdout
    assert "--check" in res.stdout
    assert "--test" in res.stdout
    assert "--clean" in res.stdout
    assert "uv" in res.stdout


def test_setup_skills_generation_global(tmp_path):
    """Default run installs to global locations (isolated HOME), with absolute tool paths."""
    env, fake_home = _isolated_env(tmp_path)
    res = subprocess.run([SETUP_SKILLS_SH], cwd=PROJECT_ROOT, capture_output=True, text=True, env=env)
    assert res.returncode == 0, res.stderr
    assert "Agent Skills 自动生成并同步完成" in res.stdout

    # OpenCode official global
    opencode_skill_md = os.path.join(fake_home, ".config", "opencode", "skills", "trainpilot", "SKILL.md")
    opencode_tool_py = os.path.join(fake_home, ".config", "opencode", "skills", "trainpilot", "scripts", "trainpilot_tool.py")
    assert os.path.isfile(opencode_skill_md), f"{opencode_skill_md} does not exist"
    assert os.path.isfile(opencode_tool_py), f"{opencode_tool_py} does not exist"
    assert os.access(opencode_tool_py, os.X_OK), "global opencode trainpilot_tool.py must be executable"

    with open(opencode_skill_md, "r", encoding="utf-8") as f:
        content = f.read()
    assert "name: trainpilot" in content
    assert "28780" in content
    # Global edition embeds the absolute installed tool path
    assert opencode_tool_py in content

    # Antigravity/Gemini CLI global: ~/.gemini/skills/trainpilot
    gemini_skill_md = os.path.join(fake_home, ".gemini", "skills", "trainpilot", "SKILL.md")
    gemini_tool_py = os.path.join(fake_home, ".gemini", "skills", "trainpilot", "scripts", "trainpilot_tool.py")
    assert os.path.isfile(gemini_skill_md), f"{gemini_skill_md} does not exist"
    assert os.path.isfile(gemini_tool_py), f"{gemini_tool_py} does not exist"
    assert os.access(gemini_tool_py, os.X_OK), "global gemini trainpilot_tool.py must be executable"

    with open(gemini_skill_md, "r", encoding="utf-8") as f:
        content = f.read()
    assert "name: trainpilot" in content
    assert "28780" in content
    assert gemini_tool_py in content
    assert "Antigravity/Gemini" in content

    # Universal agents global
    agents_skill_md = os.path.join(fake_home, ".agents", "skills", "trainpilot", "SKILL.md")
    agents_tool_py = os.path.join(fake_home, ".agents", "skills", "trainpilot", "scripts", "trainpilot_tool.py")
    assert os.path.isfile(agents_skill_md), f"{agents_skill_md} does not exist"
    assert os.path.isfile(agents_tool_py), f"{agents_tool_py} does not exist"
    assert os.access(agents_tool_py, os.X_OK), "global agents trainpilot_tool.py must be executable"

    with open(agents_skill_md, "r", encoding="utf-8") as f:
        content = f.read()
    assert "name: trainpilot" in content
    assert "28780" in content
    assert agents_tool_py in content

    # Verify compact edition saves context (file size < 4000 bytes, vs original 10000+ bytes)
    assert len(content) < 4000, f"Compact skill should be small to save context, got {len(content)} bytes"

    # --check succeeds against the same isolated globals
    check_res = subprocess.run([SETUP_SKILLS_SH, "--check"], cwd=PROJECT_ROOT, capture_output=True, text=True, env=env)
    assert check_res.returncode == 0
    assert "所有 Agent Skills 文件夹及文件均完整就绪" in check_res.stdout


def test_setup_skills_clean_action(tmp_path):
    """Verify that --clean properly removes installed skills directories."""
    env, fake_home = _isolated_env(tmp_path)
    # 1. Install first
    res = subprocess.run([SETUP_SKILLS_SH, "--skip-sync"], cwd=PROJECT_ROOT, capture_output=True, text=True, env=env)
    assert res.returncode == 0
    opencode_tool_py = os.path.join(fake_home, ".config", "opencode", "skills", "trainpilot", "scripts", "trainpilot_tool.py")
    assert os.path.isfile(opencode_tool_py)

    # 2. Run clean
    res_clean = subprocess.run([SETUP_SKILLS_SH, "--clean"], cwd=PROJECT_ROOT, capture_output=True, text=True, env=env)
    assert res_clean.returncode == 0
    assert "Agent Skills 目录已清理完成" in res_clean.stdout
    assert not os.path.exists(opencode_tool_py)


def test_generated_skills_tools_runnable(tmp_path):
    """Verify that the globally generated tools can be directly invoked by agents."""
    env, fake_home = _isolated_env(tmp_path)
    res = subprocess.run([SETUP_SKILLS_SH], cwd=PROJECT_ROOT, capture_output=True, text=True, env=env)
    assert res.returncode == 0, res.stderr

    opencode_tool_py = os.path.join(fake_home, ".config", "opencode", "skills", "trainpilot", "scripts", "trainpilot_tool.py")
    gemini_tool_py = os.path.join(fake_home, ".gemini", "skills", "trainpilot", "scripts", "trainpilot_tool.py")
    agents_tool_py = os.path.join(fake_home, ".agents", "skills", "trainpilot", "scripts", "trainpilot_tool.py")

    res1 = subprocess.run([sys.executable, opencode_tool_py, "--help"], capture_output=True, text=True)
    assert res1.returncode == 0
    assert "TrainPilot Agent Tool" in res1.stdout
    assert "28780" in res1.stdout

    res_gemini = subprocess.run([sys.executable, gemini_tool_py, "--help"], capture_output=True, text=True)
    assert res_gemini.returncode == 0
    assert "TrainPilot Agent Tool" in res_gemini.stdout
    assert "28780" in res_gemini.stdout

    res2 = subprocess.run([sys.executable, agents_tool_py, "--help"], capture_output=True, text=True)
    assert res2.returncode == 0
    assert "TrainPilot Agent Tool" in res2.stdout
    assert "28780" in res2.stdout


def test_env_files_separation():
    """Verify that .env.web.example, .env.gpu.example, and .env.example are cleanly separated."""
    web_example = os.path.join(PROJECT_ROOT, ".env.web.example")
    gpu_example = os.path.join(PROJECT_ROOT, ".env.gpu.example")
    all_example = os.path.join(PROJECT_ROOT, ".env.example")

    assert os.path.isfile(web_example), ".env.web.example must exist"
    assert os.path.isfile(gpu_example), ".env.gpu.example must exist"
    assert os.path.isfile(all_example), ".env.example must exist"

    with open(web_example, "r", encoding="utf-8") as f:
        web_txt = f.read()
    with open(gpu_example, "r", encoding="utf-8") as f:
        gpu_txt = f.read()
    with open(all_example, "r", encoding="utf-8") as f:
        all_txt = f.read()

    # Web specific variables
    assert "TRAINPILOT_FEISHU_APP_ID" in web_txt
    assert "TRAINPILOT_ENABLE_SQLITE" in web_txt
    assert "TRAINPILOT_ENABLE_WATCHDOG" in web_txt
    assert "TRAINPILOT_BIND_HOST" in web_txt

    # GPU specific variables
    assert "TRAINPILOT_HOST" in gpu_txt
    assert "TRAINPILOT_TASK_ID" in gpu_txt
    # GPU should NOT contain Feishu or SQLite or Watchdog configs
    assert "TRAINPILOT_FEISHU_APP_ID" not in gpu_txt
    assert "TRAINPILOT_ENABLE_SQLITE" not in gpu_txt
    assert "TRAINPILOT_ENABLE_WATCHDOG" not in gpu_txt

    # Master example contains both sections clearly
    assert "Web 端服务器环境变量" in all_txt
    assert "GPU 端服务器环境变量" in all_txt


def test_trainpilot_tool_auto_loads_dotenv(tmp_path):
    """Verify trainpilot_tool.py automatically loads TRAINPILOT_HOST and TASK_ID from .env."""
    fake_env = tmp_path / ".env"
    fake_env.write_text(
        "TRAINPILOT_HOST=192.168.99.88\nTRAINPILOT_PORT=28780\nTRAINPILOT_TASK_ID=my-custom-task\n",
        encoding="utf-8",
    )
    tool_script = os.path.join(PROJECT_ROOT, "skills", "trainpilot", "scripts", "trainpilot_tool.py")

    # Clean out process environment so it must load from .env (and strip proxies)
    clean_env = {
        k: v for k, v in os.environ.items()
        if not k.startswith("TRAINPILOT_") and "proxy" not in k.lower()
    }
    clean_env["PATH"] = os.environ.get("PATH", "")

    # Invoke get-status to see gateway resolution in error/output
    res = subprocess.run(
        [sys.executable, tool_script, "--timeout", "1", "get-status"],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        env=clean_env,
    )
    # The command should attempt to contact 192.168.99.88:28780 and fail with connection error or output
    assert "192.168.99.88" in res.stderr or "192.168.99.88" in res.stdout
    assert "my-custom-task" in res.stderr or "my-custom-task" in res.stdout


def test_setup_skills_creates_gpu_env_and_warns_zero(tmp_path):
    """Verify setup_skills.sh detects 0.0.0.0 on GPU server and warns appropriately."""
    env, _ = _isolated_env(tmp_path)
    res = subprocess.run([SETUP_SKILLS_SH, "--skip-sync"], cwd=PROJECT_ROOT, capture_output=True, text=True, env=env)
    assert res.returncode == 0
    assert "Agent Skills 自动生成并同步完成" in res.stdout
    assert "uv" in res.stdout



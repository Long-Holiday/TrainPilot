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
    """Verify start.sh --help outputs usage instructions."""
    res = subprocess.run([START_SH, "--help"], cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert res.returncode == 0
    assert "TrainPilot Web 服务启动脚本使用说明" in res.stdout
    assert "--daemon" in res.stdout
    assert "28780" in res.stdout
    assert "setup_skills.sh" in res.stdout


def test_setup_skills_script_help():
    """Verify setup_skills.sh --help outputs usage instructions."""
    res = subprocess.run([SETUP_SKILLS_SH, "--help"], cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert res.returncode == 0
    assert "TrainPilot Agent Skills 生成脚本使用说明" in res.stdout
    assert "--check" in res.stdout
    assert "--project" in res.stdout
    assert "--clean" in res.stdout


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

    # --check succeeds against the same isolated globals
    check_res = subprocess.run([SETUP_SKILLS_SH, "--check"], cwd=PROJECT_ROOT, capture_output=True, text=True, env=env)
    assert check_res.returncode == 0
    assert "所有 Agent Skills 文件夹及文件均完整就绪" in check_res.stdout


def test_setup_skills_project_scope(tmp_path):
    """--project keeps the legacy project-level layout working."""
    env, _ = _isolated_env(tmp_path)
    res = subprocess.run([SETUP_SKILLS_SH, "--project"], cwd=PROJECT_ROOT, capture_output=True, text=True, env=env)
    assert res.returncode == 0, res.stderr

    opencode_skill_md = os.path.join(PROJECT_ROOT, ".opencode", "skills", "trainpilot", "SKILL.md")
    opencode_tool_py = os.path.join(PROJECT_ROOT, ".opencode", "skills", "trainpilot", "scripts", "trainpilot_tool.py")
    assert os.path.isfile(opencode_skill_md), f"{opencode_skill_md} does not exist"
    assert os.path.isfile(opencode_tool_py), f"{opencode_tool_py} does not exist"

    with open(opencode_skill_md, "r", encoding="utf-8") as f:
        content = f.read()
    assert ".opencode/skills/trainpilot/scripts/trainpilot_tool.py" in content

    gemini_skill_md = os.path.join(PROJECT_ROOT, ".gemini", "skills", "trainpilot", "SKILL.md")
    gemini_tool_py = os.path.join(PROJECT_ROOT, ".gemini", "skills", "trainpilot", "scripts", "trainpilot_tool.py")
    assert os.path.isfile(gemini_skill_md), f"{gemini_skill_md} does not exist"
    assert os.path.isfile(gemini_tool_py), f"{gemini_tool_py} does not exist"

    with open(gemini_skill_md, "r", encoding="utf-8") as f:
        content = f.read()
    assert ".gemini/skills/trainpilot/scripts/trainpilot_tool.py" in content


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

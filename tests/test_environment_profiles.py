import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_tilelang_is_opt_in_for_hopper_hosts():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())

    default_dependencies = project["project"]["dependencies"]
    hopper_dependencies = project["project"]["optional-dependencies"]["hopper"]

    assert not any(dep.startswith("tilelang") for dep in default_dependencies)
    assert hopper_dependencies == ["tilelang==0.1.9"]

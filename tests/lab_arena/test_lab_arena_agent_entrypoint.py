from __future__ import annotations

import json

import pytest

from lab_arena import agent_entrypoint


def _paths(tmp_path, harness: str):
    source = tmp_path / "source"
    source.mkdir()
    (source / "harness.py").write_text(harness, encoding="utf-8")
    input_path = tmp_path / "icp.json"
    input_path.write_text(
        json.dumps({"icp": {"icp_id": "arena:test", "prompt": "find leads"}}),
        encoding="utf-8",
    )
    return source, input_path, tmp_path / "companies.json"


def test_entrypoint_calls_a_reexported_run_icp_and_wraps_the_list(tmp_path):
    source, input_path, output_path = _paths(
        tmp_path, "from implementation import run_icp\n"
    )
    (source / "implementation.py").write_text(
        "def run_icp(icp):\n    return [{'company_name': icp['prompt']}]\n",
        encoding="utf-8",
    )
    agent_entrypoint.run(
        source_dir=source, input_path=input_path, output_path=output_path
    )
    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "companies": [{"company_name": "find leads"}]
    }


@pytest.mark.parametrize(
    "definition",
    [
        "async def run_icp(icp):\n    return []\n",
        "def run_icp():\n    return []\n",
        "def run_icp(icp, other):\n    return []\n",
        "def run_icp(icp, *args):\n    return []\n",
        "def run_icp(icp, **kwargs):\n    return []\n",
        "def run_icp(icp, *, option=None):\n    return []\n",
    ],
)
def test_entrypoint_requires_exactly_one_synchronous_positional_input(
    tmp_path, definition
):
    source, input_path, output_path = _paths(tmp_path, definition)
    with pytest.raises(agent_entrypoint.AgentContractError):
        agent_entrypoint.run(
            source_dir=source, input_path=input_path, output_path=output_path
        )
    assert not output_path.exists()


def test_entrypoint_requires_a_list_of_company_objects(tmp_path):
    source, input_path, output_path = _paths(
        tmp_path, "def run_icp(icp):\n    return {'companies': []}\n"
    )
    with pytest.raises(agent_entrypoint.AgentContractError, match="must return a list"):
        agent_entrypoint.run(
            source_dir=source, input_path=input_path, output_path=output_path
        )

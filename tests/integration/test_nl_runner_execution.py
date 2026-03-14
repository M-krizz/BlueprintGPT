from pathlib import Path
from uuid import uuid4

from nl_interface.runner import execute_response
from nl_interface.service import process_user_request


def test_execute_response_runs_algorithmic_and_writes_artifacts():
    response = process_user_request(
        "Need 2 bedrooms, 1 bathroom, 1 kitchen and 1 drawing room on a 10 marla plot with a north entrance and minimize corridor.",
        resolution={"boundary_size": (15, 10)},
    )
    work_dir = Path("outputs") / f"test_nl_runner_integration_{uuid4().hex}"

    execution = execute_response(
        response,
        output_dir=str(work_dir),
        output_prefix="integration_nl_algo",
    )

    assert execution["status"] == "completed"
    assert execution["backend_target"] == "algorithmic"
    assert Path(execution["artifact_paths"]["svg"]).exists()
    assert Path(execution["artifact_paths"]["report"]).exists()
    assert execution.get("design_score") is not None
    assert execution.get("alternatives")
    assert 1 <= len(execution["alternatives"]) <= 3
    for alt in execution["alternatives"]:
        assert alt.get("design_score") is not None
        assert set(alt.keys()) >= {"strategy_name", "design_score"}

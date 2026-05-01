"""Smoke test for the ${name} scaffold."""

from ${name_snake} import run


def test_run_round_trips_payload() -> None:
    assert run(payload={"x": 1}) == {"echo": {"x": 1}}

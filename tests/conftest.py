import pytest


@pytest.fixture(autouse=True)
def _run_in_tmp_dir(tmp_path, monkeypatch):
    """Training writes run folders under ./runs; keep them out of the repo."""
    monkeypatch.chdir(tmp_path)

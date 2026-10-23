"""The dashboard contract of run_tests.sh: with DUCK_TESTS_LAST_RUN set, the conftest reporter
writes <path>.md and <path>.json with totals, per-file counts, failures and skip reasons.
Checked by running pytest on a tiny throwaway test file under tests/ (so tests/conftest.py loads).
"""
import json
import os
import subprocess
import sys

import pytest

from conftest import PY, ROOT

TESTS_DIR = os.path.join(ROOT, "tests")
PROBE = os.path.join(TESTS_DIR, "_probe_reporter_tmp.py")       # created and removed by the fixture
PROBE_SRC = '''
import pytest
def test_ok():
    assert True
def test_bad():
    assert 1 == 2, "deliberate"
@pytest.mark.skip(reason="deliberate skip")
def test_skipped():
    pass
'''


@pytest.fixture
def probe_run(tmp_path):
    out = tmp_path / "probe_last_run.md"
    with open(PROBE, "w") as fh:
        fh.write(PROBE_SRC)
    try:
        env = dict(os.environ, DUCK_TESTS_LAST_RUN=str(out))
        r = subprocess.run([PY, "-m", "pytest", PROBE, "-q", "-p", "no:cacheprovider"], cwd=ROOT,
                           capture_output=True, text=True, timeout=120, env=env)
    finally:
        os.remove(PROBE)
    return out, r


def test_reporter_writes_md_and_json(probe_run):
    out, r = probe_run
    assert r.returncode == 1, r.stdout[-800:]            # one deliberate failure
    assert out.exists(), "last_run.md not written"
    js = out.with_suffix(".json")
    assert js.exists(), "last_run.json not written"
    rep = json.load(open(js))
    assert rep["totals"] == {"passed": 1, "failed": 1, "skipped": 1}
    assert rep["exit"] == 1
    (fname, counts), = rep["files"].items()
    assert fname.endswith("_probe_reporter_tmp.py")
    assert counts["passed"] == 1 and counts["failed"] == 1 and counts["skipped"] == 1
    assert rep["failures"][0]["test"].endswith("::test_bad") and "deliberate" in rep["failures"][0]["output"]
    assert rep["skipped"][0]["reason"] == "deliberate skip"
    md = out.read_text()
    assert "1 passed, 1 failed, 1 skipped" in md
    assert "## Failures" in md and "test_bad" in md
    assert "## Skipped" in md and "deliberate skip" in md


def test_reporter_is_silent_without_the_env_var(tmp_path):
    with open(PROBE, "w") as fh:
        fh.write(PROBE_SRC)
    try:
        env = {k: v for k, v in os.environ.items() if k != "DUCK_TESTS_LAST_RUN"}
        before = set(os.listdir(TESTS_DIR))
        subprocess.run([PY, "-m", "pytest", PROBE, "-q", "-p", "no:cacheprovider"], cwd=ROOT,
                       capture_output=True, text=True, timeout=120, env=env)
        after = set(os.listdir(TESTS_DIR))
    finally:
        os.remove(PROBE)
    new = {f for f in after - before if f.startswith("last_run")}
    assert not new, f"an ad-hoc pytest run must not overwrite the dashboard's last_run files: {new}"


def test_run_tests_sh_exists_and_parses():
    sh = os.path.join(ROOT, "run_tests.sh")
    assert os.access(sh, os.X_OK), "run_tests.sh must be executable"
    r = subprocess.run(["bash", "-n", sh], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    text = open(sh).read()
    assert "DUCK_TESTS_LAST_RUN" in text and "JAX_PLATFORMS=cpu" in text and "CUDA_VISIBLE_DEVICES" in text

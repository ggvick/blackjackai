import os
import pathlib
import subprocess
import sys

import pytest


@pytest.mark.slow
def test_notebook_executes_with_nbmake():
    pytest.importorskip("nbmake")
    notebook_path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "notebooks"
        / "Blackjack_RL_Mega_Notebook.ipynb"
    )
    assert notebook_path.exists(), "Expected mega notebook to exist"
    env = dict(**os.environ, NBMAKE_ACTIVE="1")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--nbmake", str(notebook_path)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    assert "passed" in result.stdout

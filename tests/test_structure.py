from pathlib import Path


def test_project_files_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / "src" / "chandraquant" / "pipeline.py").exists()
    assert (root / "requirements.txt").exists()
    assert (root / "README.md").exists()


def test_pipeline_defines_run_experiment():
    source = Path(__file__).resolve().parents[1] / "src" / "chandraquant" / "pipeline.py"
    text = source.read_text(encoding="utf-8")
    assert "def run_experiment()" in text

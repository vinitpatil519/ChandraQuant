"""Run the ChandraQuant / Graha-Sūcana experiment from the project root."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chandraquant.pipeline import run_experiment


if __name__ == "__main__":
    run_experiment()

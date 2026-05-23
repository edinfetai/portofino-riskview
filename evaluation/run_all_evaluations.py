from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    PROJECT_ROOT / "evaluation" / "evaluate_ml.py",
    PROJECT_ROOT / "evaluation" / "evaluate_nlp.py",
    PROJECT_ROOT / "evaluation" / "evaluate_vision.py",
]


def main() -> None:
    for script in SCRIPTS:
        print("\n" + "=" * 72)
        print(f"Starte: {script.name}")
        print("=" * 72)
        subprocess.run([sys.executable, str(script)], check=True)


if __name__ == "__main__":
    main()

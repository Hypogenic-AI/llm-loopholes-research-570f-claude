"""Run all three analyses in sequence and produce a consolidated summary."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd):
    print(f"\n$ {' '.join(cmd)}")
    p = subprocess.run(cmd, cwd=ROOT)
    return p.returncode


def main():
    py = sys.executable
    run([py, "src/analyze.py"])
    run([py, "src/glm.py"])
    run([py, "src/analyze_choi.py"])
    run([py, "src/analyze_selfaware.py"])
    run([py, "src/qualitative_examples.py"])
    print("\nAll analyses complete.")


if __name__ == "__main__":
    main()

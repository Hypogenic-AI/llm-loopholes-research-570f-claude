"""Wrapper: run KCL for one model (called by run_all.sh in parallel)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_kcl import run_one_model

if __name__ == "__main__":
    model_short = sys.argv[1]
    seeds = tuple(int(s) for s in sys.argv[2].split(",")) if len(sys.argv) > 2 else (0, 1, 2)
    only = int(sys.argv[3]) if len(sys.argv) > 3 else None
    run_one_model(model_short, seeds=seeds, only=only)

"""Start native PX4 thrust/allocator training with learned rate torque."""
from pathlib import Path
import sys

# Prefer this checkout over another editable installation, including worktrees.
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from rate_rl.launcher import main


if __name__ == "__main__":
    raise SystemExit(main(project=PROJECT))

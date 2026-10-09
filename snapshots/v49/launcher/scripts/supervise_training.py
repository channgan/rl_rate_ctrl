"""Project operational entry; live work requires all supervision gates."""
from pathlib import Path
import sys
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from rate_rl.startup_entry import main
if __name__ == '__main__':raise SystemExit(main())

import os
import sys
from pathlib import Path

# Run as a file (e.g. an editor's Run button) rather than `python -m corners`:
# make the package importable from the project root.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Must be set before numpy loads: its default thread pool can fail to
# allocate memory on some Windows machines.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# Team names include non-Latin-1 characters (e.g. Turkish clubs), which the
# default Windows console encoding cannot print. Line buffering keeps output
# ordered when it is redirected to a file by a scheduler.
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

from corners.cli import main  # noqa: E402

sys.exit(main())

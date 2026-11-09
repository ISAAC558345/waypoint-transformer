"""Formal evaluation command. Importing this module does not load torch."""
from __future__ import annotations

from wt.evaluation.registered_episode.run import main


if __name__ == "__main__":
    raise SystemExit(main())

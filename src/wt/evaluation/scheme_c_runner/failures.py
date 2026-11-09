"""Index the bundled historical baseline log without external Git objects."""
from __future__ import annotations

import re

from wt.evaluation.scheme_c_runner.protocol import FAILURE_LOG_COMMIT, FAILURE_LOG_LOCAL_PATH, FAILURE_LOG_PATH

FORMAL_MARKERS = (
    "phase6iaa_formal_planner",
    "run_active_episode",
    "phase6f_system_bringup",
    "bit_queue_v2",
    "test_phase6iaa_formal_planner",
    "test_phase6b_bit_queue",
)


def _log_text() -> str:
    return FAILURE_LOG_LOCAL_PATH.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")


def _exception_line(block: str) -> str:
    for line in reversed(block.splitlines()):
        stripped = line.strip()
        if not stripped or stripped.startswith(("File ", "Traceback", "---")):
            continue
        if re.match(r"[A-Za-z_][A-Za-z0-9_.]*(Error|Exception)", stripped):
            return stripped
    return ""


def _impact(block: str, kind: str) -> str:
    if any(marker in block for marker in FORMAL_MARKERS):
        return "current_formal_path"
    if kind == "missing_file":
        return "missing_file"
    if kind == "historical_state":
        return "historical_state"
    return "assertion_failure"


def build_failure_index() -> dict:
    text = _log_text()
    rows = []
    for block in re.split(r"\n={10,}\n", text):
        match = re.match(r"(FAIL|ERROR): (.+)", block.strip())
        if match is None:
            continue
        message = _exception_line(block)
        if message.startswith("FileNotFoundError"):
            kind = "missing_file"
        elif message.startswith("RuntimeError"):
            kind = "historical_state"
        elif message.startswith("AssertionError"):
            kind = "assertion_failure"
        else:
            kind = "unclassified"
        rows.append(
            {
                "status": match.group(1),
                "test": match.group(2).split(" (", 1)[0],
                "kind": kind,
                "impact": _impact(block, kind),
                "message": message[:240],
            }
        )
    counts = {}
    for row in rows:
        counts[row["kind"]] = counts.get(row["kind"], 0) + 1
    current = [row for row in rows if row["impact"] == "current_formal_path"]
    footer = ""
    for line in text.splitlines():
        if line.startswith("Ran ") or line.startswith("FAILED"):
            footer = line if not footer else footer + " | " + line
    return {
        "source_commit": FAILURE_LOG_COMMIT,
        "source_path": FAILURE_LOG_PATH,
        "unittest_footer": footer,
        "signature_count": len(rows),
        "counts": counts,
        "current_formal_path_count": len(current),
        "current_formal_path": current,
        "rows": rows,
    }
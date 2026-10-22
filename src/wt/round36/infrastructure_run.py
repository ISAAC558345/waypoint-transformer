"""Run the project-local temp and spawn smoke audit before scientific work."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile

from .infrastructure import TMP_RUNTIME, configure_project_tempdir, process_smoke
from .run import _write_json


def main() -> None:
    target = configure_project_tempdir("runtime")
    parent = Path(tempfile.mkdtemp(prefix="round36_parent_"))
    probe = parent / "probe.txt"; probe.write_text("ok", encoding="utf-8")
    parent_ok = probe.read_text(encoding="utf-8") == "ok"
    shutil.rmtree(parent); cleanup_ok = not parent.exists()
    results: dict[str, object] = {}
    for workers in (2, 4):
        try:
            rows = process_smoke(workers)
            results[str(workers)] = {"passed": all(index == expected and valid and path == str(TMP_RUNTIME) for expected, (index, valid, path) in enumerate(rows)), "rows": rows}
        except Exception as error:  # audit the exact engineering failure without hiding it
            results[str(workers)] = {"passed": False, "error_type": type(error).__name__, "error": str(error)}
    passed = parent_ok and cleanup_ok and all(row.get("passed") for row in results.values())
    audit = {"schema_version": "round36_windows_tempdir_audit_v1", "temp_dir": str(target), "parent_process_temp_ok": parent_ok, "child_process_temp_ok": all(row.get("passed") for row in results.values()), "concurrent_temp_creation": results, "cleanup_ok": cleanup_ok, "permission_error_resolved": passed, "passed": passed}
    _write_json("round36_windows_tempdir_audit.json", audit)
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()

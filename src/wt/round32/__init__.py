"""Read-only Round-32 effective-horizon semantics audit."""


def run_audit():
    """Lazily expose the audit without pre-importing its executable module."""
    from .audit import run_audit as _run_audit

    return _run_audit()


__all__ = ["run_audit"]

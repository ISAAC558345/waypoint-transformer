"""Round-34 frozen benchmark-complexity audit package."""

# Keep package import side-effect free so ``python -m wt.legacy.benchmark_complexity.audit`` runs
# exactly the selected audit module and never creates an implicit warm cache.
__all__: list[str] = []
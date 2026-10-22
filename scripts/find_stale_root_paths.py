import re
from pathlib import Path

pat = re.compile(r"""(?:ROOT|root|parents\[\d+\])\s*/\s*["']([^"']+)["']""")
skip = (
    "docs/",
    "results/",
    "src/",
    "round_",
    "final_artifacts",
    "configs/",
    "backup/",
    "experiments/",
    "tests/",
    "scripts/",
    "paper_assets",
    "AGENTS",
    "LICENSE",
    "README",
    "requirements",
    "environment",
    "CURRENT_",
    "ROUND_",
    "round_manifest",
)
for path in Path("src").rglob("*.py"):
    parts = path.parts
    if "wt" in parts:
        nxt = parts[parts.index("wt") + 1]
        if nxt.startswith("round"):
            continue
    text = path.read_text(encoding="utf-8", errors="ignore")
    for match in pat.finditer(text):
        name = match.group(1)
        if name.startswith(skip):
            continue
        if name.endswith((".md", ".csv", ".json", ".png", ".txt")):
            print(f"{path.as_posix()}: {name}")

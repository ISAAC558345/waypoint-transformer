import re
from pathlib import Path

root = Path("docs/archive")
wanted = {
    "LLV2_PROTOCOL.md",
    "LLV2_REWARD_DRAFT.md",
    "LLV2_SAFETY_PROJECTOR_REPORT.md",
    "level2_causal_validation_report.md",
    "planner_audit_report.md",
    "PLANS.md",
    "FINAL_README_CN.md",
    "FINAL_RESEARCH_SUMMARY_CN.md",
    "FINAL_PROJECT_MANIFEST.md",
}
out = root / "phase_reports"
found = set()
marker = re.compile(r"^## 源文件 `([^`]+)`\s*$", re.M)
for path in (root / "merged").glob("*.md"):
    text = path.read_text(encoding="utf-8")
    matches = list(marker.finditer(text))
    for index, match in enumerate(matches):
        name = match.group(1)
        if name not in wanted:
            continue
        start = match.end()
        stop = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        (out / name).write_text(text[start:stop].strip() + "\n", encoding="utf-8")
        found.add(name)
print("extracted", sorted(found))
print("missing", sorted(wanted - found))

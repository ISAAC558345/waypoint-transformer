"""R1A reads a copied artifact root. It does not load a model or start an episode."""
from __future__ import annotations

import hashlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from wt.legacy_formal_repro_r1a.export import main as export_main
from wt.legacy_formal_repro_r1a.inspect import main as inspect_main
from wt.legacy_formal_repro_r1a.read import build_view

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "legacy_formal_repro_r1"
REPORT = ROOT / "docs" / "research" / "Legacy_Formal_Repro_R1_Report_CN.md"
CONFIGS = (
    "322000_scheme_c.json",
    "322000_g1_rate.json",
    "322001_scheme_c.json",
    "322001_g1_rate.json",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_package(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE / "run_manifest.json", dest / "run_manifest.json")
    config_dir = dest / "configs"
    config_dir.mkdir()
    for name in CONFIGS:
        shutil.copy2(SOURCE / "configs" / name, config_dir / name)


class LegacyFormalReproR1ATests(unittest.TestCase):
    def test_copied_root_resolves_configs_without_the_absolute_path(self) -> None:
        reader = ROOT / "src" / "wt" / "legacy_formal_repro_r1a"
        for path in reader.glob("*.py"):
            for line in path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                self.assertFalse(stripped.startswith("import torch") or stripped.startswith("from torch"))
                self.assertNotIn("torch.load", stripped)
        with tempfile.TemporaryDirectory() as folder:
            package = Path(folder)
            _copy_package(package)
            view = build_view(package)
            self.assertEqual(len(view["arms"]), 4)
            for arm in view["arms"]:
                resolved = (package / arm["config_relative"]).resolve()
                self.assertTrue(resolved.is_file())
                self.assertEqual(_sha256(resolved), arm["config_sha256"])
                self.assertNotEqual(str(resolved), arm["original_path"])
                self.assertTrue(str(resolved).startswith(str(package.resolve())))

    def test_inspect_and_export_leave_the_saved_bytes_unchanged(self) -> None:
        watched = [SOURCE / "run_manifest.json", REPORT]
        watched.extend(SOURCE / "episodes" / f"{scenario}_{policy}.json" for scenario, policy in (
            (322000, "scheme_c"),
            (322000, "g1_rate"),
            (322001, "scheme_c"),
            (322001, "g1_rate"),
        ))
        watched.extend(SOURCE / "configs" / name for name in CONFIGS)
        before = {path: _sha256(path) for path in watched}
        stdout = io.StringIO()
        original = sys.stdout
        sys.stdout = stdout
        try:
            self.assertEqual(inspect_main(["--artifact-root", str(SOURCE)]), 0)
        finally:
            sys.stdout = original
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "view"
            self.assertEqual(export_main(["--artifact-root", str(SOURCE), "--output", str(output)]), 0)
            with self.assertRaises(FileExistsError):
                export_main(["--artifact-root", str(SOURCE), "--output", str(output)])
        completed = subprocess.run(
            [sys.executable, "-c", "import sys; sys.path.insert(0, 'src'); from wt.legacy_formal_repro_r1.run import cli; "
             "raise SystemExit(0 if cli([]) == 0 and cli(['--finalize']) == 2 and cli(['--rerun']) == 2 else 1)"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        after = {path: _sha256(path) for path in watched}
        self.assertEqual(after, before)

    def test_derived_view_repeats_the_saved_192_step_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            package = Path(folder)
            _copy_package(package)
            view = build_view(package)
        saved = json.loads((SOURCE / "run_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(view["source_environment_steps"], 192)
        self.assertEqual(view["source_environment_steps"], int(saved["environment_steps"]))
        self.assertFalse(view["source_steps_are_a_new_run"])
        self.assertEqual(view["reader_environment_steps"], 0)
        self.assertEqual(view["reader_model_loads"], 0)
        saved_payloads = {
            (int(arm["scenario_id"]), arm["policy"]): arm["payload_bits"] for arm in saved["arms"]
        }
        self.assertEqual(
            {(arm["scenario_id"], arm["policy"]): arm["payload_bits"] for arm in view["arms"]},
            saved_payloads,
        )
        self.assertEqual(len(view["pairs"]), 2)
        for pair, saved_pair in zip(view["pairs"], saved["pairs"]):
            self.assertEqual(pair["payload_difference_bits"], saved_pair["payload_difference_bits"])
            self.assertEqual(pair["recomputed_payload_difference_bits"], saved_pair["payload_difference_bits"])
        self.assertEqual(view["missing_original_summaries"], ["results/phase6if_ranking_loss/summary.json"])


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Fail-fast audit for the FREUID frozen repository and optional local public-base CSV."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "Documents for submission" / "Technical report"
REPLY = ROOT / "Documents for submission" / "Kaggle submission reply template" / "REPLY_TEMPLATE.txt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def module_constant(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise ValueError(f"{name} not found in {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-draft", action="store_true",
                        help="permit PENDING report/reply fields during local preparation")
    parser.add_argument("--public-base", type=Path,
                        help="optional ignored local frozen public-base CSV to hash")
    args = parser.parse_args()

    manifest = json.loads((ROOT / "FROZEN_MANIFEST.json").read_text(encoding="utf-8"))
    expected_names = tuple(Path(model["path"]).name for model in manifest["models"])
    expected_weights = tuple(manifest["weights"])
    entrypoint = ROOT / "docker" / "prepare_submission.py"
    if module_constant(entrypoint, "MODEL_NAMES") != expected_names:
        raise ValueError("Docker MODEL_NAMES differs from frozen manifest")
    if tuple(module_constant(entrypoint, "MODEL_WEIGHTS")) != expected_weights:
        raise ValueError("Docker MODEL_WEIGHTS differs from frozen manifest")

    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    first_instruction = next(
        line.strip() for line in dockerfile.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if not first_instruction.startswith("FROM ") or "@sha256:" not in first_instruction:
        raise ValueError("Docker base is not pinned by immutable digest")
    for model in manifest["models"]:
        path = ROOT / model["path"]
        if not path.is_file() or path.stat().st_size != model["bytes"]:
            raise ValueError(f"missing or wrong-size checkpoint: {path}")
        actual = sha256(path)
        if actual != model["sha256"]:
            raise ValueError(f"checkpoint hash mismatch: {path}")
        if f"COPY {model['path']}" not in dockerfile or actual not in dockerfile:
            raise ValueError(f"Dockerfile does not copy and verify {model['path']}")
        print(f"PASS checkpoint {path.name} {actual}")

    report_tex = (REPORT_DIR / "freuid_technical_report.tex").read_text(encoding="utf-8")
    reply_text = REPLY.read_text(encoding="utf-8")
    if not args.allow_draft and ("PENDING" in report_tex or "PENDING" in reply_text):
        raise ValueError("report/reply still contains PENDING fields")
    report_pdf = REPORT_DIR / "freuid_technical_report.pdf"
    if not report_pdf.is_file() or report_pdf.stat().st_size < 10_000:
        raise ValueError("compiled technical report PDF is missing or implausibly small")
    print(f"PASS report PDF {report_pdf.stat().st_size} bytes {sha256(report_pdf)}")

    tracked = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    forbidden_prefixes = ("the-freuid-challenge-2026-ijcai-ecai/", "external_data/")
    leaked = [name for name in tracked if name.replace("\\", "/").startswith(forbidden_prefixes)]
    if leaked:
        raise ValueError(f"restricted dataset paths are tracked: {leaked[:3]}")
    if "real_recap_eval.csv" in tracked:
        raise ValueError("official row metadata remains tracked; commit its deletion")
    print("PASS no competition/external image tree is tracked")

    if args.public_base:
        expected = manifest["frozen_public_base"]
        if sha256(args.public_base) != expected["sha256"]:
            raise ValueError("local public-base hash differs from manifest")
        print(f"PASS public base {expected['rows']} rows {expected['sha256']}")

    if not (ROOT / "LICENSE").is_file() or "MIT License" not in (ROOT / "LICENSE").read_text():
        raise ValueError("MIT source license is missing")
    print("PREFLIGHT PASS" + (" (draft fields allowed)" if args.allow_draft else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

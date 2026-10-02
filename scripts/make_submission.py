"""Build the zip to deliver in Campus Virtual.

Only whitelisted paths are included, so dev files (.github, .venv, .idea, docs/...)
never end up in the delivery. It also checks that requirements.txt only contains
the packages allowed by the assignment.

Usage:
    python scripts/make_submission.py
    python scripts/make_submission.py --with-tests --with-datasets --with-results
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dist" / "iml-work2-code.zip"

ALWAYS = ["src", "experiments", "README.md", "requirements.txt"]
OPTIONAL = {
    "with_tests": "tests",
    "with_datasets": "datasets",
    "with_results": "results",
}
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache", ".venv", ".idea"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}

ALLOWED_PACKAGES = {
    "arff-loader",
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "matplotlib",
    "seaborn",
}


def check_requirements(path: Path) -> list[str]:
    """Return the packages in requirements.txt that are not allowed."""
    bad = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "-")):
            continue
        match = re.match(r"[A-Za-z0-9_.\-]+", line)
        if not match:
            continue
        name = match.group(0).lower().replace("_", "-")
        if name not in ALLOWED_PACKAGES:
            bad.append(name)
    return bad


def iter_files(entry: Path):
    if entry.is_file():
        yield entry
        return
    for path in sorted(entry.rglob("*")):
        if not path.is_file():
            continue
        if EXCLUDED_PARTS & set(path.relative_to(ROOT).parts):
            continue
        if path.suffix in EXCLUDED_SUFFIXES:
            continue
        yield path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag, folder in OPTIONAL.items():
        parser.add_argument(
            f"--{flag.replace('_', '-')}", action="store_true", help=f"include {folder}/"
        )
    args = parser.parse_args()

    entries = list(ALWAYS) + [folder for flag, folder in OPTIONAL.items() if getattr(args, flag)]

    missing = [e for e in ALWAYS if not (ROOT / e).exists()]
    if missing:
        print(f"ERROR: missing required paths: {', '.join(missing)}")
        return 1

    bad = check_requirements(ROOT / "requirements.txt")
    if bad:
        print(
            "ERROR: requirements.txt has packages not allowed by the assignment: " + ", ".join(bad)
        )
        return 1

    OUT.parent.mkdir(exist_ok=True)
    count = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for entry in entries:
            target = ROOT / entry
            if not target.exists():
                print(f"warning: {entry} does not exist, skipped")
                continue
            for path in iter_files(target):
                zf.write(path, path.relative_to(ROOT))
                count += 1
    print(f"Created {OUT.relative_to(ROOT)} with {count} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())

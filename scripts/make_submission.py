"""Build the ZIP to deliver in Campus Virtual (layout of "How to deliver IML code and report"):

    submission.zip
    ├── report.pdf
    └── code/
        ├── README.md
        ├── requirements.txt
        ├── src/ experiments/ ...
        └── datasets/

Only files tracked by git are packed, so local files (caches, venv, IDE settings, exploratory
scripts, old results) never end up in the delivery. Before writing the ZIP it checks that
requirements.txt only lists the allowed packages and that every packed Python file only imports
the standard library, those packages or our own modules.

Usage (from the project root):
    python scripts/make_submission.py --report path/to/report.pdf
    python scripts/make_submission.py --report report.pdf --with-results --with-tests
    python scripts/make_submission.py --draft          # no report yet, for a dry run
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dist" / "submission.zip"
DEFAULT_REPORT = ROOT / "report" / "report.pdf"

# Always inside code/: everything needed to install and run every experiment.
ALWAYS = ["README.md", "requirements.txt", "src", "experiments", "datasets"]
OPTIONAL = {"with_tests": "tests", "with_results": "results"}

ALLOWED_PACKAGES = {
    "arff-loader",
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "matplotlib",
    "seaborn",
}
# Import names of the allowed packages.
ALLOWED_IMPORTS = {"arff_loader", "numpy", "scipy", "pandas", "sklearn", "matplotlib", "seaborn"}
# Development tools that only the tests may import (not in requirements.txt).
TEST_ONLY_IMPORTS = {"pytest"}


def tracked_files(entry: str) -> list[Path]:
    """Files under ``entry`` tracked by git (paths relative to the project root)."""
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", entry],
        cwd=ROOT,
        capture_output=True,
        check=True,
    )
    names = [name for name in result.stdout.decode().split("\0") if name]
    return [Path(name) for name in names if (ROOT / name).is_file()]


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


def local_modules(files: list[Path]) -> set[str]:
    """Top-level names our own code can be imported as (modules and packages)."""
    names = set()
    for path in files:
        if path.suffix != ".py":
            continue
        parts = path.parts
        # src/<module>.py, src/<package>/..., experiments/<module>.py, tests/<module>.py
        if len(parts) >= 2 and parts[0] in {"src", "experiments", "tests", "scripts"}:
            names.add(Path(parts[1]).stem)
        names.add(parts[0])
    return names


def bad_imports(files: list[Path]) -> list[str]:
    """``file: module`` for every import outside the stdlib, the allowed packages and our code."""
    allowed = set(sys.stdlib_module_names) | ALLOWED_IMPORTS | local_modules(files)
    problems = []
    for path in files:
        if path.suffix != ".py":
            continue
        try:
            tree = ast.parse((ROOT / path).read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as error:
            problems.append(f"{path}: syntax error at line {error.lineno}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules = [node.module]
            else:
                continue
            for module in modules:
                top = module.split(".")[0]
                if top in allowed or (path.parts[0] == "tests" and top in TEST_ONLY_IMPORTS):
                    continue
                problems.append(f"{path}: {module}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--report", type=Path, default=DEFAULT_REPORT, help="PDF placed at the zip root"
    )
    parser.add_argument(
        "--draft", action="store_true", help="build without report.pdf (not for delivery)"
    )
    parser.add_argument("--with-tests", action="store_true", help="include tests/ (needs pytest)")
    parser.add_argument(
        "--with-results", action="store_true", help="include the committed results/"
    )
    parser.add_argument(
        "--out", type=Path, default=OUT, help="output zip (default: dist/submission.zip)"
    )
    args = parser.parse_args(argv)

    report = args.report if args.report.is_absolute() else Path.cwd() / args.report
    if not report.is_file() and not args.draft:
        print(f"ERROR: report not found: {report}. Pass --report <pdf> (or --draft for a dry run).")
        return 1

    missing = [entry for entry in ALWAYS if not (ROOT / entry).exists()]
    if missing:
        print(f"ERROR: missing required paths: {', '.join(missing)}")
        return 1

    bad = check_requirements(ROOT / "requirements.txt")
    if bad:
        print(
            "ERROR: requirements.txt has packages not allowed by the assignment: " + ", ".join(bad)
        )
        return 1

    entries = list(ALWAYS) + [folder for flag, folder in OPTIONAL.items() if getattr(args, flag)]
    files = [path for entry in entries for path in tracked_files(entry)]
    untracked = [entry for entry in entries if not tracked_files(entry)]
    if untracked:
        print(f"ERROR: nothing tracked by git under: {', '.join(untracked)} (commit them first)")
        return 1

    problems = bad_imports(files)
    if problems:
        print("ERROR: imports not allowed by the assignment:\n  " + "\n  ".join(problems))
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as zf:
        if report.is_file():
            zf.write(report, "report.pdf")
        for path in files:
            zf.write(ROOT / path, Path("code") / path)
    note = "" if report.is_file() else " (DRAFT: no report.pdf, not ready to deliver)"
    print(f"Created {args.out} with {len(files)} code files{note}")
    if args.with_tests:
        print("Note: tests/ needs pytest, which is a development tool (not in requirements.txt).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Create a new project from this template, or import the template into an existing directory.

Copies every file tracked by git in this template except the template-only files below, renames
the project, and records the template commit in `.template-source`. Existing files in the target
are never overwritten: if any would be, nothing is copied and the conflicts are listed.

Usage (standard library only, Python >= 3.9):

    python3 scripts/new_project.py <target_dir> [--name NAME] [--dry-run]

See TEMPLATE.md (section '导入后要改什么') for what to change afterwards.
"""

from __future__ import annotations

import argparse
import datetime
import re
import shutil
import subprocess
import sys
from pathlib import Path

TEMPLATE_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_NAME = "lightning-hydra-template"

# files that describe or maintain the template itself and do not belong in a new project
TEMPLATE_ONLY = {
    "TEMPLATE.md",
    "scripts/new_project.py",
    "docs/solutions.md",
}

# (file, pattern, replacement): the project name in the new project; each pattern must match once
RENAMES = [
    ("pyproject.toml", r'^name = "lightning-hydra-template"$', 'name = "{name}"'),
    ("environment.yaml", r"^name: lightning-hydra-template$", "name: {name}"),
    (
        "configs/logger/wandb.yaml",
        r'^  project: "lightning-hydra-template"$',
        '  project: "{name}"',
    ),
    ("README.md", r"^# Lightning-Hydra-Template$", "# {name}"),
    ("README.md", r"^conda activate lightning-hydra-template$", "conda activate {name}"),
]

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def git(*args: str) -> str:
    """Run a git command in the template repository and return its stdout."""
    return subprocess.run(
        ["git", "-C", str(TEMPLATE_ROOT), *args], check=True, capture_output=True, text=True
    ).stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("target", type=Path, help="new or existing project directory")
    parser.add_argument("--name", help="project name (default: the target directory name)")
    parser.add_argument("--dry-run", action="store_true", help="only list what would be copied")
    args = parser.parse_args()

    target = args.target.expanduser().resolve()
    name = args.name or target.name
    if not NAME_RE.match(name):
        parser.error(f"invalid project name {name!r}: use letters, digits, '.', '_' and '-'")
    if target == TEMPLATE_ROOT or TEMPLATE_ROOT in target.parents:
        parser.error("the target must be outside the template directory")

    if git("status", "--porcelain", "--untracked-files=no").strip():
        print(
            "error: the template has uncommitted changes; commit them first, so that "
            ".template-source points at the code that is copied",
            file=sys.stderr,
        )
        return 1
    commit = git("rev-parse", "HEAD").strip()
    branch = git("rev-parse", "--abbrev-ref", "HEAD").strip()

    files = sorted(f for f in git("ls-files", "-z").split("\0") if f and f not in TEMPLATE_ONLY)
    conflicts = [f for f in files + [".template-source"] if (target / f).exists()]
    if conflicts:
        print(
            f"error: {len(conflicts)} file(s) already exist in {target}, nothing was copied:",
            file=sys.stderr,
        )
        for f in conflicts:
            print(f"  {f}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(f"would copy {len(files)} files from {TEMPLATE_ROOT} ({commit[:7]}) to {target}:")
        for f in files:
            print(f"  {f}")
        return 0

    # check the renames against the template before writing anything
    contents: dict[str, str] = {}
    for file, pattern, replacement in RENAMES:
        text = contents.get(file) or (TEMPLATE_ROOT / file).read_text(encoding="utf-8")
        new_text, count = re.subn(pattern, replacement.format(name=name), text, flags=re.M)
        if count != 1:
            print(
                f"error: expected one match of {pattern!r} in {file}, found {count}; "
                "update RENAMES in scripts/new_project.py",
                file=sys.stderr,
            )
            return 1
        contents[file] = new_text

    for f in files:
        dst = target / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        if f in contents:
            dst.write_text(contents[f], encoding="utf-8")
            shutil.copymode(TEMPLATE_ROOT / f, dst)
        else:
            shutil.copy2(TEMPLATE_ROOT / f, dst)

    try:
        remote = git("remote", "get-url", "origin").strip()
    except subprocess.CalledProcessError:
        remote = "(none)"
    (target / ".template-source").write_text(
        f"template: {TEMPLATE_NAME}\n"
        f"remote: {remote}\n"
        f"branch: {branch}\n"
        f"commit: {commit}\n"
        f"date: {datetime.date.today().isoformat()}\n",
        encoding="utf-8",
    )

    print(f"copied {len(files)} files from {TEMPLATE_NAME} ({branch} {commit[:7]}) to {target}")
    print(f"project name: {name} (pyproject.toml, environment.yaml, wandb project, README)")
    print(f"next steps: section '导入后要改什么' in {TEMPLATE_ROOT / 'TEMPLATE.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

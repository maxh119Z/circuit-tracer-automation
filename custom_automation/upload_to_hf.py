"""
Upload folders from custom_automation/ to the HF dataset
`circuit-tracer-automation/custom_automation`.

Usage:
    # Interactive — lists folders in custom_automation/, prompts for selection + dest
    python custom_automation/upload_to_hf.py

    # Upload a local folder to a specific path in the repo
    python custom_automation/upload_to_hf.py \
        --local custom_automation/artifacts \
        --dest gemma-2/wikipedia/test_graphs

    # Upload the contents of a folder (not the folder itself) into a dest path
    python custom_automation/upload_to_hf.py \
        --local custom_automation/artifacts --dest gemma-2/wikipedia/test_graphs --flatten

Auth: run `python huggingface_login.py <token>` first, or set HF_TOKEN env var,
or pass --token. Get a write token at https://huggingface.co/settings/tokens.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, login
from huggingface_hub.errors import RepositoryNotFoundError

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent

DEFAULT_REPO_ID = "circuit-tracer-automation/custom_automation"
DEFAULT_REPO_TYPE = "dataset"

# Folders we never want to push.
SKIP_NAMES = {"__pycache__", ".DS_Store", ".ipynb_checkpoints"}
IGNORE_PATTERNS = ["__pycache__/*", "*.pyc", ".DS_Store", ".ipynb_checkpoints/*", "*.bak"]


def list_uploadable_folders(root: Path) -> list[Path]:
    """Return all non-skipped subdirectories of root, sorted alphabetically."""
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name not in SKIP_NAMES)


def pick_folder_interactive(root: Path) -> Path:
    """Prompt the user to choose one of root's subdirectories interactively."""
    folders = list_uploadable_folders(root)
    if not folders:
        sys.exit(f"No folders found in {root}")
    print(f"\nFolders in {root.relative_to(REPO_ROOT)}/:\n")
    for i, f in enumerate(folders, 1):
        print(f"  [{i}] {f.name}")
    print()
    raw = input("Pick a folder number to upload: ").strip()
    try:
        return folders[int(raw) - 1]
    except (ValueError, IndexError):
        sys.exit(f"Invalid selection: {raw!r}")


def ensure_repo(api: HfApi, repo_id: str, repo_type: str) -> None:
    """Verify the HF repo exists, offering to create it if not."""
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
    except RepositoryNotFoundError:
        create = input(f"Repo {repo_id!r} ({repo_type}) not found. Create it? [y/N]: ").strip().lower()
        if create != "y":
            sys.exit("Aborted.")
        api.create_repo(repo_id=repo_id, repo_type=repo_type, private=True, exist_ok=True)
        print(f"Created {repo_type} repo: {repo_id}")


def upload(local: Path, dest: str, repo_id: str, repo_type: str, api: HfApi, large: bool) -> None:
    """Upload local folder to dest path inside the HF repo."""
    dest_clean = dest.strip("/")
    print(f"\nUploading {local} -> {repo_id}/{dest_clean or '(root)'}  ({repo_type}){'  [large]' if large else ''}")
    if large:
        # upload_large_folder is resumable + parallel; best for multi-GB uploads.
        # It doesn't accept path_in_repo, so we stage by symlinking if needed.
        if dest_clean:
            sys.exit(
                "--large mode uploads to the repo root. Either:\n"
                f"  (a) drop --large and use upload_folder (works for 14GB but slower/less resumable), or\n"
                f"  (b) rearrange files locally so {local} already mirrors the dest path and upload from a staging dir."
            )
        api.upload_large_folder(
            folder_path=str(local),
            repo_id=repo_id,
            repo_type=repo_type,
            ignore_patterns=IGNORE_PATTERNS,
        )
    else:
        api.upload_folder(
            folder_path=str(local),
            path_in_repo=dest_clean or None,
            repo_id=repo_id,
            repo_type=repo_type,
            ignore_patterns=IGNORE_PATTERNS,
            commit_message=f"Upload {local.name} -> {dest_clean or '/'}",
        )
    url_prefix = "datasets/" if repo_type == "dataset" else ""
    print(f"Done: https://huggingface.co/{url_prefix}{repo_id}/tree/main/{dest_clean}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--local", help="Local folder to upload (path relative to repo root or absolute)")
    parser.add_argument("--dest", help="Destination path inside the HF repo, e.g. gemma-2/wikipedia/test_graphs")
    parser.add_argument("--flatten", action="store_true",
                        help="Upload contents of --local into --dest directly (default: upload folder into --dest/<name>)")
    parser.add_argument("--repo", default=DEFAULT_REPO_ID, help=f"HF repo (default: {DEFAULT_REPO_ID})")
    parser.add_argument("--repo-type", default=DEFAULT_REPO_TYPE, choices=["dataset", "model", "space"])
    parser.add_argument("--token", help="HF token (else uses cached login or HF_TOKEN env var)")
    parser.add_argument("--large", action="store_true",
                        help="Use upload_large_folder (resumable/parallel; ONLY uploads to repo root — ignores --dest)")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if token:
        login(token=token, add_to_git_credential=False)

    api = HfApi()
    try:
        whoami = api.whoami()
        print(f"Authenticated as: {whoami['name']}")
    except Exception as e:
        sys.exit(f"Not authenticated. Run `python huggingface_login.py <token>` first.\n  ({e})")

    # Resolve local folder
    if args.local:
        local = Path(args.local)
        if not local.is_absolute():
            local = (REPO_ROOT / local).resolve()
        if not local.is_dir():
            sys.exit(f"Local folder not found: {local}")
    else:
        local = pick_folder_interactive(PACKAGE_DIR)

    # Resolve destination in repo
    dest = args.dest
    if dest is None:
        default_dest = local.name
        dest = input(f"Destination path in repo [{default_dest}]: ").strip() or default_dest

    # If not flattening, append the folder name so the folder itself lands at dest/<name>
    if not args.flatten:
        dest = f"{dest.strip('/')}/{local.name}" if dest.strip("/") else local.name

    ensure_repo(api, args.repo, args.repo_type)
    upload(local, dest, args.repo, args.repo_type, api, large=args.large)


if __name__ == "__main__":
    main()
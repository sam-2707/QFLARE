Workspace Organization Summary
=============================

This small utility groups the paper files into a `paper/` directory and archives
LaTeX auxiliary files into `paper/archive/` to keep the repository root tidy.

Files added
- `scripts/organize_workspace.ps1` — PowerShell script that performs a dry-run by default and performs actual moves when called with `-Execute`.

How it works
- Dry-run (default): shows what would be moved and logs actions to `organize.log`.
- Execute mode: moves listed paper files into `paper/` and moves common LaTeX aux files into `paper/archive/`.

Usage

1. Dry-run (recommended first):

    .\scripts\organize_workspace.ps1

2. Execute (actually move files):

    .\scripts\organize_workspace.ps1 -Execute

Notes and safety
- The script is conservative; it moves only a small list of files. It does not delete files.
- Review `organize.log` after running to confirm changes.

If you'd like me to move additional directories (e.g., figures, build artifacts, or other assets), tell me which ones and I will expand the script and run it.

Windows exe cross-build

This repository includes a GitHub Actions workflow to build a Windows executable using PyInstaller on a Windows runner.

How to run

1. Commit and push the repository to GitHub.
2. In the repository Actions tab, run the "Build Windows exe" workflow (workflow_dispatch).
3. After the run completes, download the artifact named `shogigame-windows-exe` which contains `shogigame.exe`.

Notes

- The workflow runs on `windows-latest` and installs `pygame` and `pyinstaller`.
- If additional binary dependencies are required, add them to `requirements.txt` or the workflow install step.

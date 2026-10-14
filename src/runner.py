import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_tool(tool: str, *args: str | Path) -> None:
    # Each tool is a Pixi task defined only in its own feature, so Pixi picks the environment.
    command = [
        "pixi",
        "run",
        tool,
        *(str(arg) for arg in args),
    ]

    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
    )


import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TOOL_ENVIRONMENTS = {
    "preprocess": "preprocess",
    "detect": "detect",
    "annotate": "annotate",
}


def run_tool(tool: str, *args: str | Path) -> None:
    if tool not in TOOL_ENVIRONMENTS:
        raise ValueError(f"Unknown tool: {tool}")

    command = [
        "pixi",
        "run",
        "--environment",
        TOOL_ENVIRONMENTS[tool],
        "python",
        str(PROJECT_ROOT / "src" / "toolbox" / f"{tool}.py"),
        *(str(arg) for arg in args),
    ]

    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
    )


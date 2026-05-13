"""Small helper for appending structured research-log entries."""

from __future__ import annotations

from datetime import date
from pathlib import Path


def append_research_log(
    log_path: Path,
    title: str,
    worked: list[str],
    did_not_work: list[str],
    todo: list[str],
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "",
        f"## {date.today().isoformat()} - {title}",
        "",
        "### Worked",
        "",
    ]
    lines.extend(f"- {item}" for item in worked or ["No new successes recorded."])
    lines.extend(["", "### Did Not Work / Caveats", ""])
    lines.extend(f"- {item}" for item in did_not_work or ["No new failures recorded."])
    lines.extend(["", "### Left To Do", ""])
    lines.extend(f"- {item}" for item in todo or ["No new follow-up items recorded."])
    lines.append("")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


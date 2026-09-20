"""Base helpers shared by the commands that list projects."""

import os
from datetime import datetime, timezone
from enum import Enum


class SortBy(str, Enum):
    alpha = "alpha"
    date = "date"


def scan_projects(projects_path: str, sort_by: SortBy, desc: bool) -> list[os.DirEntry]:
    """Return the sorted project folders under ``projects_path``.

    An empty list means there is nothing to list - the folder is missing or holds
    no project.
    """
    if not os.path.exists(projects_path):
        return []

    entries = [e for e in os.scandir(projects_path) if e.is_dir()]

    if sort_by == SortBy.date:
        entries.sort(key=lambda e: e.stat().st_ctime, reverse=desc)
    else:
        entries.sort(key=lambda e: e.name, reverse=desc)

    return entries


def format_created(entry: os.DirEntry) -> str:
    """Return the folder creation time as ``YYYY-MM-DD HH:MM:SS`` in UTC."""
    return datetime.fromtimestamp(entry.stat().st_ctime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def has_content(project_path: str, part_dir: str) -> bool:
    """Tell whether ``part_dir`` exists inside the project and holds at least one entry."""
    part_path = os.path.join(project_path, part_dir)
    if not os.path.isdir(part_path):
        return False

    with os.scandir(part_path) as entries:
        return any(entries)


def project_row(index: int, plain_cells: list[str], display_cells: list[str], is_current: bool) -> list[str]:
    """Return the table cells of one project row.

    The current project is printed plain in bold yellow, with the counter also
    underlined. Every other row keeps the markup of ``display_cells``.
    """
    if is_current:
        return [f"[bold underline yellow]{index}[/bold underline yellow]"] + [
            f"[bold yellow]{cell}[/bold yellow]" for cell in plain_cells
        ]

    return [str(index)] + list(display_cells)

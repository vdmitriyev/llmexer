"""Helpers to compress a generated experiment database into a 7z archive.

An ``experiment_*.db`` holds a rendered prompt for every combination and the full
response text and JSON for every row that ran, so it compresses very well. The
archive keeps the database under its own file name, which makes it a drop-in
replacement for moving the ``.db`` around.
"""

import os

import py7zr


def compact_db_to_7z(db_path: str, archive_path: str) -> tuple[int, int]:
    """Write ``db_path`` into ``archive_path`` as a 7z archive.

    The database is stored under its own file name, so extracting the archive
    anywhere recreates ``experiment_<YYYYMMDD>_<NN>.db``.

    Returns ``(source_bytes, archive_bytes)``.
    """

    with py7zr.SevenZipFile(archive_path, "w") as archive:
        archive.write(db_path, arcname=os.path.basename(db_path))

    return os.path.getsize(db_path), os.path.getsize(archive_path)

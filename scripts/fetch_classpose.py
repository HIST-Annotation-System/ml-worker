"""Downloads Classpose's source into vendor/classpose, where pyproject.toml expects it.

Classpose is not on PyPI, and a regular install of it leaves out its subpackages (its build configuration lists only
the top-level package), so it is installed from this copy in editable mode instead. Run before `uv sync`.
"""

import io
import shutil
import tarfile
import urllib.request
from pathlib import Path

# Pinned: the worker relies on what this commit's `eval` returns. After changing it, run this and `uv lock` again.
COMMIT = "ea7eeca74a767d3bf48fcc7b1d7a800c21ce7c9b"
URL = f"https://github.com/sohmandal/classpose/archive/{COMMIT}.tar.gz"
TARGET = Path(__file__).resolve().parent.parent / "vendor" / "classpose"
# Only what the install needs; the rest of the repository is experiments and a QuPath extension.
WANTED = ("pyproject.toml", "README.md", "LICENSE", "src/")


def main() -> None:
    marker = TARGET / ".commit"
    if marker.exists() and marker.read_text().strip() == COMMIT:
        return
    shutil.rmtree(TARGET, ignore_errors=True)
    with urllib.request.urlopen(URL) as response:
        archive = tarfile.open(fileobj=io.BytesIO(response.read()), mode="r:gz")
    with archive:
        for member in archive.getmembers():
            # Every path starts with the archive's own top-level directory.
            _, _, path = member.name.partition("/")
            if path.startswith(WANTED):
                member.name = path
                archive.extract(member, TARGET, filter="data")
    marker.write_text(COMMIT + "\n")


if __name__ == "__main__":
    main()

"""Portable provenance labels for generated files."""
from pathlib import Path


def path_label(path, root):
    """Use a repository-relative label when possible; retain external paths."""
    path = Path(path).resolve()
    try:
        return path.relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return str(path)

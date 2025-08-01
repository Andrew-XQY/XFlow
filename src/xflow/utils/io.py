"""Input/Output utilities for file operations."""

import shutil
from pathlib import Path
from typing import List, Optional, Union

from .typing import PathLikeStr


def copy_file(
    source_path: PathLikeStr, target_path: PathLikeStr, filename: Optional[str] = None
) -> Path:
    """Copy a file to target directory.

    Args:
        source_path: Source file path
        target_path: Target directory path
        filename: Optional new filename (uses source filename if None)

    Returns:
        Path to the copied file
    """
    source_path = Path(source_path)
    target_path = Path(target_path)

    target_path.mkdir(parents=True, exist_ok=True)
    target_filename = filename or source_path.name
    target_path = target_path / target_filename

    shutil.copy2(source_path, target_path)
    return target_path


def create_directory(path: PathLikeStr) -> Path:
    """Create directory if it doesn't exist.

    Args:
        path: Directory path to create

    Returns:
        Path to the created directory
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def scan_files(
    root_paths: Union[PathLikeStr, List[PathLikeStr]],
    extensions: Optional[Union[str, List[str]]] = None,
    return_type: str = "path",
    recursive: bool = True,
) -> Union[List[str], List[Path]]:
    """
    Scan directories for files with specified extensions.

    Args:
        root_paths: Single path or list of paths to scan
        extensions: File extensions to include (e.g., '.jpg' or ['.jpg', '.png']).
                   If None, includes all files.
        return_type: "path" to return Path objects, "str" to return strings
        recursive: Whether to scan subdirectories recursively

    Returns:
        Sorted list of file paths
    """
    # Normalize inputs
    if isinstance(root_paths, (str, Path)):
        paths = [Path(root_paths)]
    else:
        paths = [Path(p) for p in root_paths]

    if extensions is None:
        ext_set = None
    elif isinstance(extensions, str):
        ext_set = {extensions.lower()}
    else:
        ext_set = {ext.lower() for ext in extensions}

    # Collect files
    file_paths = []
    for root_path in paths:
        if not root_path.exists():
            continue

        pattern = "**/*" if recursive else "*"
        for file_path in root_path.glob(pattern):
            if file_path.is_file():
                if ext_set is None or file_path.suffix.lower() in ext_set:
                    if return_type in ["str", "string"]:
                        file_paths.append(str(file_path))
                    else:
                        file_paths.append(file_path)

    return sorted(file_paths)

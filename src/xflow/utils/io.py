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


def resolve_save_path(
    directory: Optional[str] = None,
    filename: Optional[str] = None,
    extension: Optional[str] = None,
    auto_timestamp: bool = True,
) -> Path:
    """Resolve output path from various input combinations.

    Priority:
    1. directory contains full path with extension -> use directly
    2. directory + filename (with extension) -> combine
    3. directory + filename + extension -> combine all
    4. directory only (no filename) -> auto-generate with timestamp

    Args:
        directory: Output directory, OR full file path if contains extension
        filename: Output filename (optional, with or without extension)
        extension: File extension (e.g., '.png' or 'png')
        auto_timestamp: If True, auto-generate filename when not provided

    Returns:
        Resolved Path object

    Raises:
        ValueError: If path cannot be resolved

    Examples:
        >>> resolve_save_path(directory="/output/result.png")
        Path('/output/result.png')

        >>> resolve_save_path(directory="/output", filename="result.png")
        Path('/output/result.png')

        >>> resolve_save_path(directory="/output", filename="result", extension=".png")
        Path('/output/result.png')

        >>> resolve_save_path(directory="/output")  # auto timestamp
        Path('/output/20250205_123456_789012.png')
    """
    from datetime import datetime

    # Normalize extension
    if extension and not extension.startswith("."):
        extension = f".{extension}"

    # Case 1: directory is full path with extension
    if directory:
        dir_path = Path(directory)
        if dir_path.suffix:
            return dir_path

    # Need directory from here
    if not directory:
        raise ValueError("Cannot resolve save path: 'directory' is required.")

    base_dir = Path(directory)

    # Case 2/3: directory + filename
    if filename:
        file_path = Path(filename)
        if file_path.suffix:
            return base_dir / filename
        elif extension:
            return base_dir / f"{filename}{extension}"
        else:
            raise ValueError(
                f"Cannot resolve path: filename '{filename}' has no extension "
                f"and no 'extension' parameter provided."
            )

    # Case 4: auto-generate filename
    if auto_timestamp:
        from datetime import timezone

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        ext = extension or ".png"
        return base_dir / f"{timestamp}{ext}"
    else:
        raise ValueError(
            "Cannot resolve path: no filename provided and auto_timestamp=False."
        )

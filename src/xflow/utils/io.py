"""Input/Output utilities for file operations."""

import shutil
import tarfile
import tempfile
import zipfile
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

    If path appears to be a file (has an extension), creates parent directories.
    If path appears to be a directory, creates the directory itself.

    Args:
        path: Directory or file path

    Returns:
        Path to the created directory (parent if file, itself if directory)
    """
    path_obj = Path(path)

    # Check if path has a file extension (likely a file)
    if path_obj.suffix:
        # It's a file path, create parent directories
        dir_path = path_obj.parent
    else:
        # It's a directory path, create it directly
        dir_path = path_obj

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


# Zip and tar management


def resolve_resource_dir(target_dir: PathLikeStr) -> Path:
    """Resolve a resource directory by using an archive fallback.

    Behavior:
    1. If ``target_dir`` exists and is a directory, return it.
    2. If ``target_dir`` does not exist, look for same-name archives
       (``.zip``, ``.tar``, ``.tar.gz``, ``.tgz``), extract, and return directory.
    3. If neither directory nor same-name archive exists, raise ``FileNotFoundError``.

    Args:
        target_dir: Directory path to resolve

    Returns:
        Resolved directory path

    Raises:
        FileNotFoundError: If directory and supported archives are missing
        ValueError: If archive contains unsafe paths or link/device entries
    """
    target = Path(target_dir).resolve()
    if target.exists():
        if target.is_dir():
            return target
        raise NotADirectoryError(f"Path exists but is not a directory: {target}")

    archive = _find_same_name_archive(target)
    if archive is None:
        raise FileNotFoundError(
            f"Resource not found: {target} (also tried .zip/.tar/.tar.gz/.tgz)"
        )

    target.mkdir(parents=True, exist_ok=True)
    _extract_archive_to_dir_safely(archive, target)
    return target


def cleanup_resource_dir(target_dir: PathLikeStr) -> bool:
    """Remove a resolved resource directory.

    Args:
        target_dir: Directory to remove

    Returns:
        True if removed, False if missing or not a directory
    """
    target = Path(target_dir).resolve()
    if target.exists() and target.is_dir():
        shutil.rmtree(target)
        return True
    return False


def _find_same_name_archive(target_dir: Path) -> Optional[Path]:
    candidates = [
        target_dir.with_suffix(".zip"),
        Path(str(target_dir) + ".tar"),
        Path(str(target_dir) + ".tar.gz"),
        Path(str(target_dir) + ".tgz"),
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    return None


def _extract_archive_to_dir_safely(archive: Path, out_dir: Path) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_out = Path(temp_dir)

        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive, "r") as zip_file:
                _safe_extract_zip(zip_file, temp_out)
        else:
            with tarfile.open(archive, "r:*") as tar_file:
                _safe_extract_tar(tar_file, temp_out)

        normalized_source = _normalize_extracted_root(temp_out, out_dir.name)
        _move_extracted_contents(normalized_source, out_dir)


def _normalize_extracted_root(extracted_dir: Path, expected_root_name: str) -> Path:
    children = list(extracted_dir.iterdir())
    if (
        len(children) == 1
        and children[0].is_dir()
        and children[0].name == expected_root_name
    ):
        return children[0]
    return extracted_dir


def _move_extracted_contents(source_dir: Path, destination_dir: Path) -> None:
    for entry in source_dir.iterdir():
        shutil.move(str(entry), str(destination_dir / entry.name))


def _is_subpath(base_dir: Path, path: Path) -> bool:
    base = base_dir.resolve()
    target = path.resolve()
    try:
        target.relative_to(base)
        return True
    except ValueError:
        return False


def _safe_extract_zip(zip_file: zipfile.ZipFile, out_dir: Path) -> None:
    for member in zip_file.infolist():
        member_path = (out_dir / member.filename).resolve()
        if not _is_subpath(out_dir, member_path):
            raise ValueError(f"Unsafe zip member path: {member.filename}")
    zip_file.extractall(out_dir)


def _safe_extract_tar(tar_file: tarfile.TarFile, out_dir: Path) -> None:
    for member in tar_file.getmembers():
        if member.issym() or member.islnk() or member.isdev():
            raise ValueError(f"Unsafe tar member type: {member.name}")
        member_path = (out_dir / member.name).resolve()
        if not _is_subpath(out_dir, member_path):
            raise ValueError(f"Unsafe tar member path: {member.name}")
    tar_file.extractall(out_dir)

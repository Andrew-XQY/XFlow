"""General puerpose file parser."""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Type

import yaml

from .typing import PathLikeStr

SUPPORTED_FORMATS: Dict[str, Type["ConfigParser"]] = {}


def register_parser(*extensions: str):
    """
    Class decorator to register a ConfigParser under one or more file extensions.
    """

    def decorator(cls: Type["ConfigParser"]) -> Type["ConfigParser"]:
        for ext in extensions:
            key = ext.lower()
            SUPPORTED_FORMATS[key] = cls
        return cls

    return decorator


class ConfigParser(ABC):
    """Abstract base for configuration parsers."""

    @abstractmethod
    def parse(self, file_path: PathLikeStr) -> Any:
        """Parse configuration file and return data."""
        ...

    @abstractmethod
    def save(self, data: Any, file_path: PathLikeStr) -> None:
        """Save data to configuration file."""
        ...


@register_parser(".yaml", ".yml")
class YAMLParser(ConfigParser):
    """YAML configuration parser."""

    def parse(self, file_path: PathLikeStr) -> Any:
        """Parse YAML configuration file."""
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @staticmethod
    def _add_blank_lines_between_top_level_keys(yaml_text: str) -> str:
        """
        Insert a single blank line between first-level mapping entries.
        Leaves lists and nested blocks untouched.
        """
        import re

        lines = yaml_text.splitlines()
        out = []
        seen_first_key = False
        # Match a top-level key line like: key: ...   (not starting with space/#/-)
        key_re = re.compile(r'^[^\s#-][^:]*:\s*(?:#.*)?$')

        for line in lines:
            if key_re.match(line):
                if seen_first_key:
                    # ensure exactly one blank line before the next top-level key
                    if len(out) > 0 and out[-1].strip() != "":
                        out.append("")
                seen_first_key = True
            out.append(line)

        # Ensure trailing newline at EOF
        return "\n".join(out) + "\n"

    def save(self, data: Any, file_path: PathLikeStr) -> None:
        """Save data to YAML file with blank lines between top-level blocks."""
        from pathlib import Path

        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Dump to a string first
        dumped = yaml.dump(
            data,
            default_flow_style=False,
            indent=2,
            sort_keys=False,   # Preserve order
            allow_unicode=True,
        )

        # Normalize newlines and add spacing between first-level keys
        dumped = dumped.replace("\r\n", "\n").replace("\r", "\n")
        pretty = self._add_blank_lines_between_top_level_keys(dumped)

        with open(file_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(pretty)



@register_parser(".json")
class JSONParser(ConfigParser):
    """JSON configuration parser."""

    def parse(self, file_path: PathLikeStr) -> Any:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save(self, data: Any, file_path: PathLikeStr) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


def get_parser_for_file(file_path: PathLikeStr) -> ConfigParser:
    """
    Return a parser instance for the given file, based on its extension.
    Raises a ValueError listing all supported extensions if none match.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()
    parser_cls = SUPPORTED_FORMATS.get(suffix)
    if not parser_cls:
        supported = ", ".join(sorted(SUPPORTED_FORMATS.keys()))
        raise ValueError(
            f"Unsupported format '{suffix}'. " f"Supported extensions: {supported}"
        )
    return parser_cls()


def load_file(file_path: PathLikeStr) -> Any:
    """Load config file."""
    parser = get_parser_for_file(file_path)
    return parser.parse(file_path)


def save_file(data: Any, file_path: PathLikeStr) -> None:
    """Save data to file. Format determined by file extension."""
    parser = get_parser_for_file(file_path)
    parser.save(data, file_path)

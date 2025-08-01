"""Metadata collection with extensible hooks"""

import logging
from typing import Any, Dict, Mapping, Optional

from .typing import MetaHook


class MetadataCollector:
    """
    Gather metadata from components and optional hooks.

    Each instance keeps its own ordered, idempotent hook list.
    """

    def __init__(self) -> None:
        """Create a collector with no hooks and a default logger."""
        self._meta_hooks: list[MetaHook] = []
        self._logger = logging.getLogger(__name__)

    def register_meta_hook(self, fn: MetaHook) -> MetaHook:
        """
        Register (or decorate) a metadata hook.

        Hooks are called in registration order and will not be
        added twice. Returns the original function to support:

            @collector.register_meta_hook
            def my_hook(components): ...
        """
        if fn not in self._meta_hooks:
            self._meta_hooks.append(fn)
        return fn

    def remove_hook(self, fn: MetaHook) -> bool:
        """Remove a specific hook. Returns True if hook was found and removed."""
        try:
            self._meta_hooks.remove(fn)
            return True
        except ValueError:
            return False

    @property
    def hook_count(self) -> int:
        """Number of registered hooks."""
        return len(self._meta_hooks)

    def clear_hooks(self) -> None:
        """
        Remove all registered hooks.

        Useful for resetting state in tests or between runs.
        """
        self._meta_hooks.clear()

    def collect(
        self, components: Mapping[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Build a metadata dict from components + all registered hooks.

        - Calls each component’s get_metadata(), if present.
        - Runs each hook(fn), merging any returned dict.
        - Logs and records hook errors under 'hook_errors'.
        """
        result = {"config": config or {}}

        for name, comp in components.items():
            result[name] = self._collect_component_metadata(comp)

        for hook in self._meta_hooks:
            try:
                extra = hook(components)
                if isinstance(extra, dict):
                    result.update(extra)
            except Exception:
                self._logger.exception(f"hook {hook.__name__} failed")
                result.setdefault("hook_errors", []).append(hook.__name__)

        return result

    def _collect_component_metadata(self, component: Any) -> Dict[str, Any]:
        """
        Invoke component.get_metadata(), if available.

        Returns its dict or a status dict if missing/invalid/error.
        """
        component_type = type(component).__name__

        try:
            if hasattr(component, "get_metadata"):
                md = component.get_metadata()
                if isinstance(md, dict):
                    return md
                else:
                    # Handle case where get_metadata() returns non-dict
                    self._logger.warning(
                        f"{component_type}.get_metadata() returned {type(md)}, expected dict"
                    )
                    return {"type": component_type, "status": "invalid_metadata_type"}

            return {"type": component_type, "status": "no_metadata_method"}

        except Exception as e:
            self._logger.exception(f"metadata collection error for {component_type}")
            return {
                "type": component_type,
                "status": "metadata_error",
                "error": str(e),  # Include error message for debugging
            }


# Common metadata hooks
def system_info_hook(components: Mapping[str, Any]) -> Dict[str, Any]:
    """Add system information to metadata."""
    import platform
    import sys

    return {
        "system": {
            "platform": platform.platform(),
            "python_version": sys.version,
            "architecture": platform.architecture()[0],
        }
    }


def timestamp_hook(components: Mapping[str, Any]) -> Dict[str, Any]:
    """Add timestamp to metadata."""
    from datetime import datetime

    return {"timestamp": datetime.now().isoformat()}

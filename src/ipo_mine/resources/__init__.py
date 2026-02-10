# src/resources/__init__.py
"""
Resources package for the S-1 project.

Contains reference data files (e.g., JSON mappings) that can be updated over time.
Provides easy access to their file paths for both development and packaged installs.
"""

from importlib.resources import files as _pkg_files
from pathlib import Path

def get_resource_path(filename: str) -> Path:
    """
    Return the absolute path to a resource file within this package.

    Example:
        >>> from resources import get_resource_path
        >>> json_path = get_resource_path("global_section_names.json")
    """
    try:
        return Path(_pkg_files("resources") / filename)
    except Exception:
        raise FileNotFoundError(f"Resource file not found in package: {filename}")

# Commonly used resource
GLOBAL_SECTIONS_JSON = get_resource_path("global_section_names.json")

__all__ = [
    "get_resource_path",
    "GLOBAL_SECTIONS_JSON",
]

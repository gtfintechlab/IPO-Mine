"""
Utilities for the S1 project.

Currently includes:
- config: dynamic, notebook-safe path configuration
"""

from .config import (
    BASE_DIR,
    DATA_DIR,
    RAW_DIR,
    OUTPUT_DIR,
    PARSED_DIR,
    MAPPINGS_FILE,
    set_data_root,
    resolve_from_base,
    print_config,
)

from .helpers import (
    save_s1_json,
    build_user_agent,
    extract_metadata_summary
)


__all__ = [
    "BASE_DIR",
    "build_user_agent",
    "DATA_DIR",
    "RAW_DIR",
    "OUTPUT_DIR",
    "PARSED_DIR",
    "MAPPINGS_FILE",
    "set_data_root",
    "resolve_from_base",
    "print_config",
    "save_s1_json",
    "build_user_agent",
    "extract_metadata_summary"
]

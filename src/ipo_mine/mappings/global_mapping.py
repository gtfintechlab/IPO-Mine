import os
import json
from typing import Dict, List, Optional

class GlobalMapping:
    def __init__(self, path: str = "global_section_names.json"):
        self.path = path
        self.mapping: Dict[str, Dict[str, List[str]]] = self._load()

    def _load(self) -> Dict:
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def save(self) -> None:
        """Save current mapping back to disk."""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.mapping, f, indent=2, ensure_ascii=False)

    def get_canonical_name(self, key: str) -> Optional[str]:
        """Return the canonical name for a given canonical key (e.g., 'risk_factors')."""
        entry = self.mapping.get(key)
        return entry["canonical_name"] if entry else None

    def get_variants(self, key: str) -> List[str]:
        """Return the list of variants for a given canonical key."""
        entry = self.mapping.get(key)
        return entry.get("variants", []) if entry else []

    def add_variant(self, key: str, variant: str) -> None:
        """Add a new standardized variant under the given canonical key."""
        if key not in self.mapping:
            raise KeyError(f"Canonical key {key} not found in mapping")
        
        normalized = " ".join(variant.strip().split()).lower()

        if normalized not in [v.lower() for v in self.mapping[key]["variants"]]:
            self.mapping[key]["variants"].append(normalized)

    def keys(self) -> List[str]:
        """Return all canonical keys"""
        return list(self.mapping.keys())

from dataclasses import dataclass
from typing import Optional


@dataclass
class S1FilingImage:
    """Represents an image found in an S-1 filing."""
    
    img_name: str
    url: str
    local_path: Optional[str] = None
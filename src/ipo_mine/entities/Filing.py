from dataclasses import dataclass
from typing import Optional, List
from .FormType import FormType
from .FilingImage import FilingImage

@dataclass
class Filing:
    form_type: FormType
    acession_number: str
    filing_date: str
    primary_document: str
    filing_url: str

    local_path: Optional[str]
    images: List[FilingImage]
    
    cik: Optional[str] = None
    raw_content: Optional[str] = None
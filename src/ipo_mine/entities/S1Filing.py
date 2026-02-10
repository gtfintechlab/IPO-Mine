from dataclasses import dataclass
from typing import Optional, List
from .S1FilingImage import S1FilingImage


@dataclass(repr=False)
class S1Filing:
    """Represents a downloaded S-1 filing with metadata and content."""
    
    # Company identifiers
    ticker: str
    cik: str
    name: str
    
    # Filing identifiers
    accession_number: str
    filing_year: int
    filing_url: str
    primary_document: str
    
    # Company metadata
    active: bool
    sic: Optional[str]
    industry: Optional[str]
    office: Optional[str]
    exchanges: Optional[List[str]]
    
    # File information
    local_path: Optional[str]
    images: List[S1FilingImage]
    scrape_date: str
    
    # Content (added when file is read)
    raw_content: Optional[str] = None

    def __repr__(self):
        """
        Provides a concise summary including the ticker, filing year,
        a truncated URL, and the count of associated images.
        """
        
        image_count = len(self.images) if self.images is not None else 0
        
        return (
            f"S1Filing("
            f"ticker='{self.ticker}', "
            f"name='{self.name}', "
            f"year={self.filing_year}, "
            f"images={image_count}, "
            f"url='{self.filing_url}'"
            f")"
        )
from dataclasses import dataclass
from typing import Optional, List
from .Filing import Filing

@dataclass(repr=False)
class CompanyFilings:

    tickers: List[str]
    cik: str
    name: str

    sic: Optional[str]
    industry: Optional[str]
    office: Optional[str]
    exchanges: Optional[List[str]]

    filings: List[Filing]

    def __repr__(self) -> str:
        return (
            f"CompanyFilings("
            f"tickers={self.tickers}, "
            f"name='{self.name}', "
            f"cik='{self.cik}', "
            f"filings_count={len(self.filings)}"
            f")"
        )
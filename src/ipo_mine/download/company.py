import pandas as pd
import requests
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

from resources import get_resource_path

@dataclass
class Company:
    
    ticker: str
    cik: str
    active: bool

    _ticker_to_cik: Optional[Dict[str, Tuple[str, bool]]] = None
    _cik_to_ticker: Optional[Dict[str, Tuple[str, bool]]] = None
    
    @classmethod
    def _load_mapping(cls):
        """
        Loads and caches the ticker-CIK mapping from file(s) into dictionaries.
        This method is designed to run only once.
        """
        if cls._ticker_to_cik is not None and cls._cik_to_ticker is not None:
            return

        # print("Loading and caching Ticker-CIK mapping for the first time...")
        try:
            csv_path = get_resource_path("company_tickers_to_cik.csv")
            df = pd.read_csv(csv_path, dtype=str)
            df['active'] = True # Mark entries from the SEC file as active
        except FileNotFoundError:
            cls._download_ticker_cik_mapping()
            csv_path = get_resource_path("company_tickers_to_cik.csv")
            df = pd.read_csv(csv_path, dtype=str)
            df['active'] = True # Mark entries from the SEC file as active
        
        try:
            upgraded_df = pd.read_csv("upgraded_mapping.csv", dtype=str)
            upgraded_df['active'] = False # Mark entries from the custom file as inactive
            # Combine the two DataFrames. If there are duplicate tickers,
            # the one from the SEC (marked active=True) will be kept.
            df = pd.concat([upgraded_df, df]).drop_duplicates(subset=['ticker'], keep='last')
            # print("Successfully merged upgraded mapping.")
        # except FileNotFoundError:
            # print("Warning: 'upgraded_mapping' file not found. Using standard mapping only.")
        except Exception as e:
            pass
            # print(f"Warning: Could not load 'upgraded_mapping' due to an error: {e}")

        # Create fast dictionary lookups storing a tuple of (value, active_status)
        cls._ticker_to_cik = {
            str(row.ticker): (str(row.cik_str), bool(row.active))
            for row in df.itertuples(index=False)
        }

        cls._cik_to_ticker = {
            str(row.cik_str): (str(row.ticker), bool(row.active))
            for row in df.itertuples(index=False)
        }

    @classmethod
    def from_ticker(cls, ticker: str) -> 'Company':
        """Create Company from ticker, looking up the CIK and active status."""
        lookup_result = cls._lookup_cik_from_ticker(ticker.upper())
        if not lookup_result:
            raise ValueError(f"No CIK found for ticker: {ticker}")
        
        cik, active = lookup_result
        return cls(ticker=ticker.upper(), cik=cik, active=active)
    
    @classmethod
    def from_cik(cls, cik: str) -> 'Company':
        """Create Company from CIK, looking up the ticker and active status."""
        formatted_cik = cik.zfill(10)
        lookup_result = cls._lookup_ticker_from_cik(formatted_cik)
        if not lookup_result:
            return cls(ticker="", cik=formatted_cik, active=False)
            
        ticker, active = lookup_result
        return cls(ticker=ticker, cik=formatted_cik, active=active)
    
    @classmethod
    def _lookup_cik_from_ticker(cls, ticker: str) -> Optional[str]:
        """Look up CIK from the cached dictionary. 🚀"""
        cls._load_mapping()  # Ensures mapping is loaded before lookup
        return cls._ticker_to_cik.get(ticker)
    
    @classmethod
    def _lookup_ticker_from_cik(cls, cik: str) -> Optional[str]:
        """Look up ticker from the cached dictionary. 🚀"""
        cls._load_mapping() # Ensures mapping is loaded before lookup
        return cls._cik_to_ticker.get(cik)
    
    @staticmethod
    def _download_ticker_cik_mapping():
        """Download the ticker-CIK mapping from SEC."""
        headers = {"User-Agent": "Company Lookup Tool contact@example.com"}
        try:
            response = requests.get(
                "https://www.sec.gov/files/company_tickers.json", 
                headers=headers
            )
            response.raise_for_status()
            df = pd.DataFrame.from_dict(response.json(), orient="index")
            df["cik_str"] = df["cik_str"].astype(str).str.zfill(10)
    
            save_path = get_resource_path("company_tickers_to_cik.csv")
            df.to_csv(save_path, index=False)
            print(f"✅ Successfully downloaded ticker-CIK mapping → {save_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to download ticker-CIK mapping: {e}")
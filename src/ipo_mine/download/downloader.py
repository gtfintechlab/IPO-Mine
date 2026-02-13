# Use postponed evaluation of type hints so forward refs don't break at import time.
# (Annotations are stored as strings; no NameError if Company isn't imported yet.)
from __future__ import annotations

import re
import json
from pathlib import Path
from typing import Optional, TYPE_CHECKING
from enum import Enum

import pandas as pd

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from bs4 import BeautifulSoup

from dataclasses import asdict

# Project-aware default raw directory that works from notebooks too.
# RAW_DIR should point to <repo>/data/raw via utils.config logic.
from ..utils.config import RAW_DIR
from ..entities import S1Filing, S1FilingImage, CompanyFilings, FormType, Filing, FilingImage

if TYPE_CHECKING:
    # Only imported for static type checking (mypy/pyright/IDE); not at runtime.
    from company import Company

class IPODownloader:
    """
    Minimal downloader for IPO filings (S-1, S-1/A, F-1, F-1/A).

    Directory layout under a single "raw" root (download_root):
        raw/
          ├─ company_tickers_to_cik.csv      # SEC mapping (downloaded on demand)
          ├─ company_metadata.json           # simple per-ticker metadata "upsert"
          ├─ filings/<YEAR>/<TICKER>-<CIK>-<YEAR>.<ext>
          └─ images/<TICKER>/*.png (etc.)
    """

    def __init__(self, email: str, company: str, download_dir: Optional[str] = None):
        """
        Args
        ----
        email : str
            Contact email for SEC User-Agent (required by SEC fair-use policy).
        company : str
            Organization name for SEC User-Agent.
        download_dir : Optional[str]
            Root where all artifacts are written. If None, defaults to utils.config.RAW_DIR.
        """
        # SEC asks for org + email in the UA for responsible scraping.
        self.headers = {"User-Agent": f"{company} {email}".strip()}

        # Root for all outputs; default to RAW_DIR if not provided.
        self.download_root: Path = Path(download_dir).resolve() if download_dir else RAW_DIR.resolve()
        self.download_root.mkdir(parents=True, exist_ok=True)

        # Subpaths we consistently use.
        self.filings_dir: Path = self.download_root / "filings"
        self.images_dir: Path = self.download_root / "images"
        self.map_csv: Path = self.download_root / "company_tickers_to_cik.csv"
        self.meta_json: Path = self.download_root.parent / "company_metadata.json"

        self.filings_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)

        self.ensure_ticker_cik_map()

    # ---------------------------------------------------------------------
    # Helper Methods
    # ---------------------------------------------------------------------
    
    def ensure_ticker_cik_map(self) -> None:
        """
        Ensure the SEC ticker→CIK map exists locally as CSV.
        Downloads https://www.sec.gov/files/company_tickers.json and writes map_csv.
        """
        if self.map_csv.exists():
            return
        r = requests.get("https://www.sec.gov/files/company_tickers.json", headers=self.headers, timeout=30)
        r.raise_for_status()
        df = pd.DataFrame.from_dict(r.json(), orient="index")
        # SEC returns CIK as int-ish; normalize to 10 digits for EDGAR URL building.
        df["cik_str"] = df["cik_str"].astype(str).str.zfill(10)
        df.to_csv(self.map_csv, index=False)

    def lookup_ticker_from_cik(self, cik10: str) -> Optional[str]:
        """
        Get ticker symbol for a zero-padded (10-digit) CIK. Returns None if unknown.
        """
        self.ensure_ticker_cik_map()
        df = pd.read_csv(self.map_csv, dtype=str)
        row = df[df["cik_str"] == str(cik10).zfill(10)]
        return None if row.empty else row.iloc[0]["ticker"]
    
    def requests_with_exponential_backoff(
        self,
        retries=3,  # Number of retries
        backoff_factor=0.5,  # Base delay for exponential backoff (e.g., 0.5, 1, 2, 4...)
        status_forcelist=(500, 502, 503, 504),  # HTTP status codes to retry on
        allowed_methods=("GET", "POST"),  # HTTP methods to apply retries to
        raise_on_status=True,  # Whether to raise an exception for non-retryable statuses
        respect_retry_after_header=True # Respect 'Retry-After' header if present
    ):
        """
        Returns a requests.Session object configured with exponential backoff.
        """
        session = requests.Session()
        retry_strategy = Retry(
            total=retries,
            read=retries,
            connect=retries,
            backoff_factor=backoff_factor,
            status_forcelist=status_forcelist,
            allowed_methods=allowed_methods,
            raise_on_status=raise_on_status,
            respect_retry_after_header=respect_retry_after_header
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def _fetch_company_filings(self, company: Company) -> Optional[CompanyFilings]:
        
        cik = company.cik
        ticker = company.ticker

        url = f"https://data.sec.gov/submissions/CIK{cik:0>10}.json"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        response = response.json()

        sic_code = response["sic"]
        industry = response["sicDescription"]
        office = response["ownerOrg"]
        name = response["name"]
        exchanges = list(set(response["exchanges"]))

        ipo_form_types = set(["S-1/A", "S-1", "F-1/A", "F-1"])
        ipo_filings = []

        for source in [response["filings"]["recent"]] + response["filings"]["files"]:
            if "name" in source:
                url = f"https://data.sec.gov/submissions/{source['name']}"
                response_data = requests.get(url, headers=self.headers).json()
            else:
                response_data = source

            forms = response_data["form"]
            dates = response_data["filingDate"]
            acession_numbers = response_data["accessionNumber"]
            primary_documents = response_data["primaryDocument"]
            
            for i, form in enumerate(forms):
                if form in ipo_form_types:

                    if primary_documents[i]:
                        filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acession_numbers[i].replace('-', '')}/{primary_documents[i]}"
                    else:
                        filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acession_numbers[i].replace('-', '')}/{acession_numbers[i]}.txt"

                    ipo_filings.append(Filing(
                        form_type=FormType(form),
                        acession_number=response_data['accessionNumber'][i],
                        filing_date=dates[i],
                        primary_document=response_data['primaryDocument'][i],
                        filing_url=filing_url,
                        local_path=None,
                        images=[],
                        cik=str(cik)
                    ))

        if not ipo_filings:
            return None
        
        return CompanyFilings(
            tickers=[ticker],
            cik=f"{cik:0>10}",
            name=name,
            sic=sic_code,
            industry=industry,
            office=office,
            exchanges=exchanges,
            filings=ipo_filings
        )
    
    # ---------------------------------------------------------------------
    # Standard download (resolve via Company with CIK/ticker)
    # ---------------------------------------------------------------------

    def download_ipo(
        self,
        company: Company,
        download_all: bool = False,
        limit: int = 1,
        verbose: bool = False,
        save_filing: bool = True,
        save_images: bool = False,
        process_images: bool = True
    ) -> Optional[CompanyFilings]:
        
        company_filings = self._fetch_company_filings(company)
        if not company_filings:
            if verbose:
                print(f"[ERROR] No IPO filings found for {company.ticker} (CIK {company.cik})")
            return None
        
        session = self.requests_with_exponential_backoff(retries=5, backoff_factor=1)

        for i in range(min(limit, len(company_filings.filings)) if not download_all else len(company_filings.filings)):
            filing = company_filings.filings[i]
            resp = session.get(filing.filing_url, headers=self.headers, timeout=5)
            resp.raise_for_status()

            if save_filing:
                filing_year = filing.filing_date.split("-")[0]
                year_dir = self.filings_dir / filing_year
                year_dir.mkdir(parents=True, exist_ok=True)
                ext = filing.primary_document.split(".")[-1].lower() if filing.primary_document else "txt"
                filename = year_dir / f"{(company.ticker + '-') if company.ticker else ''}{company.cik}-{filing.primary_document.split('.')[0] if filing.primary_document else filing.acession_number}.{ext}"
                relative_filename = filename.resolve().relative_to(self.download_root.parent.resolve())
                filing.local_path = str(relative_filename)

                mode = "wb" if ext == "pdf" else "w"
                with open(filename, mode, encoding=None if mode == "wb" else "utf-8") as f:
                    if mode == "wb":
                        f.write(resp.content)
                    else:
                        f.write(resp.text)

                if verbose:
                    print(f"[INFO] Downloaded {company.ticker} IPO filing to {filename}")
            
            images = []
            if process_images:
                images = self.download_images_from_s1(resp.text, company.ticker, filing.filing_url, verbose=verbose, save_images=save_images)
            filing.images = images
            filing.raw_content = resp.text

        self._meta_upsert(asdict(company_filings, dict_factory=lambda x: {k: v.value if isinstance(v, Enum) else ("" if k == "raw_content" else v) for k, v in x}), key=company.cik)
        return company_filings

    # ---------------------------------------------------------------------
    # Image scraping (only meaningful for HTML filings)
    # ---------------------------------------------------------------------

    def download_images_from_s1(self, s1_text: str, ticker: str, filing_url: str, verbose: bool = False, save_images: bool = True) -> list[FilingImage]:
        """
        Parse <img> tags from an HTML filing and download them into images/<TICKER>/.

        Returns
        -------
        list[str]
            Relative paths like "./images/TICKER/<file>" (deduped and sorted).
        """
        urls = set()
        images = []

        soup = BeautifulSoup(s1_text, "html.parser")

        imgs = soup.find_all("img")
        if not imgs:
            if verbose:
                print("[INFO] No <img> tags found")
            return images

        base_url = "/".join(filing_url.split("/")[:-1]) + "/"
        out_dir = self.images_dir / ticker.upper()
        if save_images:
            out_dir.mkdir(parents=True, exist_ok=True)

        for img in imgs:
            src = img.get("src")
            if not src:
                continue
            url = base_url + src
            out_path = None

            if save_images:
                out_path = out_dir / Path(src).name
                try:
                    r = requests.get(url, headers=self.headers, timeout=30)
                    if r.status_code == 200:
                        out_path.write_bytes(r.content)
                except Exception as e:
                    if verbose:
                        print(f"[ERROR] Failed to download image {url}: {e}")

            if url not in urls:
                images.append(FilingImage(
                    img_name=src,
                    url=url,
                    local_path=str(Path("./raw/images") / ticker.upper() / out_path.name) if save_images else None
                ))
                urls.add(url)

        return images

    # ---------------------------------------------------------------------
    # Metadata JSON helpers (simple "upsert" store keyed by ticker)
    # ---------------------------------------------------------------------

    def _meta_read(self) -> dict:
        """Read company_metadata.json safely; return {} if missing or malformed."""
        if self.meta_json.exists():
            try:
                return json.loads(self.meta_json.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _meta_write(self, data: dict) -> None:
        """Write company_metadata.json pretty-printed."""
        self.meta_json.parent.mkdir(parents=True, exist_ok=True)
        self.meta_json.write_text(json.dumps(data, indent=4), encoding="utf-8")

    def _meta_upsert(self, row: dict, *, key: str) -> None:
        """Upsert a single company's metadata by ticker `key`."""
        data = self._meta_read()
        data[key] = row
        self._meta_write(data)

    def _meta_update_images(self, ticker: str, images: list[str]) -> None:
        """Attach/replace the list of downloaded images for `ticker` in metadata."""
        data = self._meta_read()
        if ticker in data:
            data[ticker]["images"] = images
            self._meta_write(data)

if __name__ == "__main__":
    # CLI demo (defaults to RAW_DIR if download_dir is omitted).
    from .company import Company  # local import to avoid import-time cycles

    downloader = S1Downloader(
        email="you@example.com",
        company="Your Organization",
        # download_dir="data/raw",  # optional override if you don't want RAW_DIR
    )

    while True:
        tkr = input("Enter a ticker or 'exit' to quit: ").strip()
        if tkr.lower() == "exit":
            break
        c = Company.from_ticker(tkr)
        downloader.download_s1(c, verbose=True, download_images=False)

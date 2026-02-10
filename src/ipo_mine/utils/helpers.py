from pathlib import Path
import json
import pandas as pd
from typing import Optional

from .config import DATA_DIR, PARSED_DIR, RAW_DIR

def load_company_metadata(path: Path | None = None) -> dict:
    """
    Load company_metadata.json as a Python dict.

    Args
    ----
    path : Path | None
        Explicit path to company_metadata.json. If None, defaults to RAW_DIR.

    Returns
    -------
    dict
        Mapping {ticker: metadata_dict}.
    """
    if path is None:
        path = DATA_DIR / "company_metadata.json"

    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def extract_metadata_summary(path: Path | None = None) -> pd.DataFrame:
    """
    Extract ticker, cik, filing_year, and filing_url from company_metadata.json.

    Args
    ----
    path : Path | None
        Optional override path.

    Returns
    -------
    pd.DataFrame
        Columns: ticker, cik, filing_year, filing_url
    """
    data = load_company_metadata(path)
    rows = []
    for tkr, meta in data.items():
        rows.append({
            "ticker": tkr,
            "cik": meta.get("cik"),
            "filing_year": meta.get("filing_year"),
            "filing_url": meta.get("filing_url"),
        })
    return pd.DataFrame(rows)

def save_s1_json(
    ticker: str,
    sections: dict,
    year: Optional[int] = None,
    parsed_dir: Path = PARSED_DIR,
) -> dict:
    """
    Create or update a JSON file for a company's S-1 filing.

    Behavior:
      - If the file does not exist, it is created with `sections`.
      - If the file exists, its contents are loaded and updated with `sections`
        (keys in `sections` overwrite existing ones, new keys are added).
      - The merged dictionary is written back to disk and returned.

    File naming convention:
        <parsed_dir>/<year>/<TICKER>_<year>_s1.json
        e.g., data/output/parsed/2019/UBER_2019_s1.json

    Args:
        ticker (str): Stock ticker symbol, e.g., "UBER".
        sections (dict): S-1 sections to add/update, e.g. {"TABLE OF CONTENTS": {...}}
        year (int, optional): Filing year. Defaults to 9999 if not provided.
        parsed_dir (Path): Root directory for parsed outputs (default: PARSED_DIR).
    """
    if not isinstance(sections, dict):
        raise TypeError("`sections` must be a dict of {section_name: section_value}.")

    # Default year handling
    year = year if year is not None else 9999

    # Ensure year-specific folder exists inside parsed_dir
    year_dir = parsed_dir / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)

    # File path for this ticker/year
    path = year_dir / f"{ticker.upper()}_{year}_s1.json"

    # Load existing or start fresh
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}

    # Merge new sections (overwrite keys if they already exist)
    data.update(sections)

    # Save back to file
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ S-1 JSON saved → {path.resolve()}")
    return data
    

def build_user_agent(company: str, email: str, download_dir: str = None) -> dict:
    """
    Construct arguments for initializing S1Downloader.

    Returns a dict with 'company' and 'email' (and optionally 'download_dir')
    so it can be passed directly via **kwargs.

    Example:
        kwargs = build_user_agent("Georgia Tech", "me@gatech.edu")
        dl = S1Downloader(**kwargs)
    """
    args = {"company": company, "email": email}
    if download_dir is not None:
        args["download_dir"] = download_dir
    return args
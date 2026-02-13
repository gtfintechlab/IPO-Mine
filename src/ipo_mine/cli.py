import argparse
import sys
from pathlib import Path
from typing import Optional

from ipo_mine.download.company import Company
from ipo_mine.download import S1Downloader
from ipo_mine.parse.ipo_parser import IPOParser
from ipo_mine.entities import Filing
from ipo_mine.resources import GLOBAL_SECTIONS_JSON
from ipo_mine.utils.config import PARSED_DIR, RAW_DIR, print_config


def command_download(args):
    """Handler for the 'download' command."""
    if not args.company and not args.ticker:
        print("Error: You must provide either a company name (--company) or ticker symbol (positional argument).")
        sys.exit(1)

    company_name = args.company
    ticker = args.ticker
    
    # If ticker is provided but company name isn't, use ticker as company name for User-Agent
    if not company_name:
        company_name = ticker

    print(f"Initializing downloader for: {company_name}")
    downloader = S1Downloader(
        email=args.email,
        company=args.org
    )
    
    # Resolve Company object
    # If ticker is provided, use it. Otherwise, we'd need a way to lookup ticker by name, 
    # but currently Company.from_ticker is the main entry point.
    if ticker:
        company_obj = Company.from_ticker(ticker)
    else:
        # Fallback: try to search? The current Company class assumes ticker is known or CIK is known.
        # For CLI simplicity, we enforce ticker for now if name resolution isn't robust.
        print("Error: Please provide the ticker symbol (e.g. SNOW).")
        sys.exit(1)

    print(f"Downloading IPO filing(s) for {company_obj.ticker}...")
    
    downloader.download_ipo(
        company_obj,
        limit=args.limit,
        verbose=args.verbose,
        save_filing=True,
        save_images=args.images,
        process_images=args.images,
        download_all=args.all
    )
    print("\nDownload complete.")


def command_parse(args):
    """Handler for the 'parse' command."""
    ticker = args.ticker
    print(f"Parsing filings for ticker: {ticker}")

    # To parse, we need a Filing object. 
    # The current IPOParser design requires a Filing object which carries the local path.
    # We can reconstruct it functionality by locating the downloaded file for this ticker.
    # However, IPOParser.parse_company is designed to iterate over the TOC and parse sections.
    # But wait, IPOParser definition:
    # def __init__(self, filing: Filing, mappings_path: str, output_base_path: str):
    #
    # It takes a SINGLE filing. 
    # To "parse a company", we probably want to find the latest S-1 for that company.
    
    # Let's use S1Downloader logic to 'fetch' the filing metadata again to get the path,
    # or just scan the RAW_DIR. 
    # Re-fetching metadata is safer to get the correct accession number / filename structure.
    
    downloader = S1Downloader(
        email="cli_user@example.com", # Email not strictly needed for local lookup if we just scan, but needed for API
        company="CLI User"
    )
    
    company_obj = Company.from_ticker(ticker)
    
    # We fetch filings metadata (fast, no download) to identify the target file
    company_filings = downloader._fetch_company_filings(company_obj)
    if not company_filings or not company_filings.filings:
        print(f"Error: No filings found for {ticker}. Have you run 'download' first?")
        sys.exit(1)

    # By default, pick the first one (latest) or iterate?
    # CLI simple mode: processed the most recent S-1.
    target_filing = company_filings.filings[0]
    
    # We need to ensure the local_path is populated.
    # The downloader populates it during download. If we just re-fetched metadata, it's None.
    # We must reconstruct the expected path.
    # Logic from downloader.py:
    # filing_year = filing.filing_date.split("-")[0]
    # filename = year_dir / f"{(company.ticker + '-') if company.ticker else ''}{company.cik}-{filing.primary_document.split('.')[0] if filing.primary_document else filing.acession_number}.{ext}"
    
    filing_year = target_filing.filing_date.split("-")[0]
    ext = target_filing.primary_document.split(".")[-1].lower() if target_filing.primary_document else "txt"
    # Construct filename
    # NOTE: Downloader logic uses ticker if available.
    filename_base = f"{ticker}-{company_obj.cik}-{target_filing.primary_document.split('.')[0] if target_filing.primary_document else target_filing.acession_number}"
    # The downloader constructs it as:
    # f"{(company.ticker + '-') if company.ticker else ''}{company.cik}-{...}"
    # Wait, let's verify exact logic in downloader.py line 222
    # filename = year_dir / f"{(company.ticker + '-') if company.ticker else ''}{company.cik}-{filing.primary_document.split('.')[0] if filing.primary_document else filing.acession_number}.{ext}"
    
    # We'll try to find the file.
    # Check if file exists at expected path.
    
    # Actually, we can just instantiate the parser with the Constructed path.
    # Access raw_dir via config
    
    relative_path = Path(filing_year) / f"{filename_base}.{ext}"
    full_path = RAW_DIR / "filings" / relative_path
    
    if not full_path.exists():
        # Try without ticker prefix if that failed (just in case logic differs)
        # But for now assume consistent logic.
        print(f"Error: Local file not found at {full_path}")
        print("Please run 'download' command first.")
        sys.exit(1)
        
    # ipo_parser uses DATA_DIR to resolve relative paths. 
    # SAW_DIR is DATA_DIR/raw. 
    # So we need path relative to DATA_DIR -> raw/filings/...
    target_filing.local_path = str(Path("raw") / "filings" / relative_path)
    
    # Initialize Parser
    parser = IPOParser(
        filing=target_filing,
        mappings_path=GLOBAL_SECTIONS_JSON,
        output_base_path=PARSED_DIR
    )
    
    # Perform parsing
    results = parser.parse_company(
        ticker=ticker,
        html_flag=not args.text, # If --text is passed, html_flag=False. Default is HTML? parse_company arg is html_flag=False default.
        validate=args.validate,
        validation_provider=args.provider,
        validation_mode=args.mode,
        validation_api_key=args.api_key
    )
    
    if not results:
        print("Parsing returned no sections.")
    else:
        print(f"Successfully parsed {len(results)} sections.")
        print(f"Results saved to: {PARSED_DIR}/{ticker}/...")


import json
import os
import getpass
from ipo_mine.parse.llm_validator import validate_section_binary, validate_section_likert

def command_validate(args):
    """Handler for the 'validate' command."""
    ticker = args.ticker
    
    # Resolve API Key
    api_key = args.api_key
    provider = args.provider
    env_var_name = f"{provider.upper()}_API_KEY"
    
    if not api_key:
        api_key = os.environ.get(env_var_name)
    
    if not api_key:
        print(f"[{provider}] API key not found in arguments or environment ({env_var_name}).")
        try:
            api_key = getpass.getpass(prompt=f"Please enter your {provider} API key: ")
        except Exception as e:
            print(f"Error reading API key: {e}")
            return

    if not api_key:
        print("Error: No API key provided. Exiting.")
        return

    try:
        from ipo_mine.download.company import Company
        company_obj = Company.from_ticker(ticker)
        cik = company_obj.cik
        print(f"Validating filings for {ticker} (CIK: {cik})")
    except Exception:
        # Fallback if lookup fails (e.g. offline)
        cik = None
        print(f"Validating filings for {ticker} (CIK lookup failed, matching by ticker string only)")

    # Identify files
    # Structure: PARSED_DIR / section_key / year / filename
    # Filename format: {identifier}_{section_key}.{ext}
    # Identifier is typically CIK{cik}_{slug} or {ticker}_{slug}
    
    matches = []
    print(f"Scanning {PARSED_DIR} for parsed files...")
    
    for root, _, files in os.walk(PARSED_DIR):
        for file in files:
            if not file.endswith('.txt'): # Validate text content
                continue
            
            # Check for match
            # 1. Check CIK match
            if cik and f"CIK{cik}" in file:
                matches.append(os.path.join(root, file))
                continue
            
            # 2. Check Ticker match (if fallback was used or explicit ticker in filename)
            # Ensure we strictly match ticker followed by _ or -
            if file.upper().startswith(f"{ticker.upper()}_") or file.upper().startswith(f"{ticker.upper()}-"):
                matches.append(os.path.join(root, file))
                continue
                
    if not matches:
        print(f"No parsed text files found for {ticker} in {PARSED_DIR}")
        return

    print(f"Found {len(matches)} files to validate.")
    
    results = []
    
    for i, file_path in enumerate(matches):
        filename = os.path.basename(file_path)
        section_name = os.path.basename(os.path.dirname(os.path.dirname(file_path))) # crude extraction from dir "section/year/"
        # Better: extract from filename suffix
        # filename is identifier_sectionkey.txt
        # But section key is reliable from directory structure
        
        print(f"[{i+1}/{len(matches)}] Validating {filename}...")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if len(content.strip()) < 50:
                print("  -> Skipped (content too short)")
                continue

            if args.mode == "binary":
                res = validate_section_binary(
                    content, 
                    section_name, 
                    provider=args.provider, 
                    api_key=api_key
                )
                answer = res.answer
                justification = res.justification
            else:
                res = validate_section_likert(
                    content, 
                    section_name, 
                    provider=args.provider, 
                    api_key=api_key
                )
                answer = res.answer
                justification = res.justification
                
            print(f"  -> Result: {answer}")
            
            results.append({
                "file": file_path,
                "section": section_name,
                "answer": answer,
                "justification": justification,
                "provider": args.provider,
                "mode": args.mode
            })
            
        except Exception as e:
            print(f"  -> Error: {e}")
            results.append({
                "file": file_path,
                "error": str(e)
            })

    # Save report
    report_dir = PARSED_DIR / "validation_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_file = report_dir / f"{ticker}_validation_report.json"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
        
    print(f"\nValidation complete. Report saved to: {report_file}")
    
    # Print summary
    total = len(results)
    if args.mode == "binary":
        yes_count = sum(1 for r in results if r.get("answer") == "Yes")
        no_count = sum(1 for r in results if r.get("answer") == "No")
        error_count = sum(1 for r in results if "error" in r)
        
        print(f"Summary: {yes_count} valid (Yes), {no_count} potentially truncated (No), {error_count} errors.")
        print(f"Total processed: {total}")
        
    elif args.mode == "likert":
        scores = [int(r.get("answer")) for r in results if isinstance(r.get("answer"), int)]
        error_count = sum(1 for r in results if "error" in r)
        if scores:
            avg = sum(scores) / len(scores)
            print(f"Summary: Average confidence score: {avg:.2f}/5 (over {len(scores)} files)")
        if error_count > 0:
            print(f"Errors encountered: {error_count}")


def main():
    parser = argparse.ArgumentParser(
        description="IPO-Mine CLI: Download and parse S-1 filings from the SEC."
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- DOWNLOAD COMMAND ---
    dl_parser = subparsers.add_parser("download", help="Download S-1 filings")
    dl_parser.add_argument("ticker", help="Company ticker symbol (e.g. SNOW)")
    dl_parser.add_argument("--company", help="Company name (optional override for User-Agent)", default="")
    dl_parser.add_argument("--email", help="Your email for SEC User-Agent", required=True)
    dl_parser.add_argument("--org", help="Your organization for SEC User-Agent", required=True)
    dl_parser.add_argument("--limit", type=int, default=1, help="Number of filings to download (default: 1)")
    dl_parser.add_argument("--all", action="store_true", help="Download all available IPO filings")
    dl_parser.add_argument("--images", action="store_true", help="Download and process images")
    dl_parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    # --- PARSE COMMAND ---
    ps_parser = subparsers.add_parser("parse", help="Parse downloaded filings")
    ps_parser.add_argument("ticker", help="Company ticker symbol to parse")
    ps_parser.add_argument("--text", action="store_true", help="Save as plain text (default is determined by parser)")
    ps_parser.add_argument("--validate", action="store_true", help="Enable LLM validation")
    ps_parser.add_argument("--provider", default="anthropic", choices=["anthropic", "openai", "google", "huggingface"], help="Validation provider")
    ps_parser.add_argument("--mode", default="binary", choices=["binary", "likert"], help="Validation mode")
    ps_parser.add_argument("--api-key", help="API key for validation provider")
    ps_parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    # --- VALIDATE COMMAND ---
    val_parser = subparsers.add_parser("validate", help="Validate existing parsed filings")
    val_parser.add_argument("ticker", help="Company ticker symbol to validate")
    val_parser.add_argument("--provider", default="anthropic", choices=["anthropic", "openai", "google", "huggingface"], help="Validation provider")
    val_parser.add_argument("--mode", default="binary", choices=["binary", "likert"], help="Validation mode")
    val_parser.add_argument("--api-key", help="API key for validation provider")

    # --- CONFIG COMMAND ---
    cfg_parser = subparsers.add_parser("config", help="Show current configuration")

    args = parser.parse_args()

    if args.command == "download":
        command_download(args)
    elif args.command == "parse":
        command_parse(args)
    elif args.command == "validate":
        command_validate(args)
    elif args.command == "config":
        print_config()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

"""
S1 Filing Parser

Provides high-level API for parsing S1 filings, including table of contents
extraction and section text extraction.
"""
import json
import re
import os
from thefuzz import process
from entities import Filing
from .toc_parser import get_parser_for_year, parse_toc_plain_text
from .section_parser import extract_section_text

from typing import Dict, Optional

class IPOParser:
    """Parser for S1 SEC filings."""

    def __init__(self, filing: Filing, mappings_path: str, output_base_path: str):
        self.filing = filing
        self.toc = self.parse_toc(filing)
        self.mappings_path = mappings_path
        self.output_base_path = output_base_path
    
    def parse_toc(self, filing: Filing | None = None) -> Dict[str, str]:
        """
        Parse the table of contents from an S1 filing.
        
        Args:
            filing: S1Filing object containing the filing data
            
        Returns:
            TOC: Dictionary mapping section names to page numbers (as strings)
        """
        if hasattr(self, 'toc'):
            return self.toc
        elif filing == None:
            print("Error: must pass in a filing object.")
            raise
        if filing.filing_url.endswith('.txt'):
            parser = parse_toc_plain_text
        else:
            if hasattr(filing, 'filing_year') and filing.filing_year:
                year = filing.filing_year
            elif hasattr(filing, 'filing_date') and filing.filing_date:
                year = filing.filing_date.split('-')[0]
            else:
                year = 2024 # Default
            parser = get_parser_for_year(year)
        raw_content = IPOParser._get_raw_content(filing)
        return parser(raw_content)
    
    def normalize_and_map_section(self, input_text: str, mappings_path: str, output_base_path: str, year: str):
        """
        Checks an input string against a list of canonical section names, manages
        directories (stratified by year), and returns the canonical key and directory path.

        Args:
            input_text: The section name string to process.
            mappings_path: The file path to the JSON file with canonical names and variants.
            output_base_path: The base directory path where section folders are stored.
            year: The filing year to stratify storage directories.
            
        Returns:
            tuple: (canonical_key, section_dir_path) or (None, None) on error.
        """
    
        try:
            with open(mappings_path, 'r', encoding='utf-8') as f:
                sections = json.load(f)
        except FileNotFoundError:
            print(f"[ERROR] Mappings file not found at: {mappings_path}")
            return None, None
        except json.JSONDecodeError:
            print(f"[ERROR] Mappings file at {mappings_path} is not valid JSON.")
            return None, None

        cleaned_input = input_text.lower().strip()
        canonical_names = {key: data['canonical_name'] for key, data in sections.items()}

        def get_or_create_path(key):
            path = os.path.join(output_base_path, key, str(year))
            if not os.path.isdir(path):
                try:
                    os.makedirs(path)
                    print(f"[INFO] New section directory created: '{path}'")
                except OSError as e:
                    print(f"[ERROR] Could not create directory at {path}: {e}")
                    return None
            return path

        for key, name in canonical_names.items():
            if cleaned_input == name:
                print(f"[INFO] Input '{input_text}' is a direct match for canonical name: '{name}'.")
                section_dir_path = get_or_create_path(key)
                return (key, section_dir_path) if section_dir_path else (None, None)

        for key, data in sections.items():
            if cleaned_input in data['variants']:
                print(f"[INFO] Input '{input_text}' is already a known variant for '{data['canonical_name']}'.")
                section_dir_path = get_or_create_path(key)
                return (key, section_dir_path) if section_dir_path else (None, None)

        best_match_name, score = process.extractOne(cleaned_input, canonical_names.values())
        if not best_match_name:
            print(f"[ERROR] Could not find any canonical names to match against.")
            return None, None
            
        best_match_key = next(key for key, name in canonical_names.items() if name == best_match_name)
        
        print(f"[ACTION] New variant found. Fuzzy matched '{input_text}' to '{best_match_name}' with score {score}.")
        sections[best_match_key]['variants'].append(cleaned_input)
        
        try:
            with open(mappings_path, 'w', encoding='utf-8') as f:
                json.dump(sections, f, indent=2)
            print(f"[ACTION] Added '{cleaned_input}' as a new variant and updated '{mappings_path}'.")
        except IOError as e:
            print(f"[ERROR] Could not write to mappings file at {mappings_path}: {e}")
            return None, None

        section_dir_path = get_or_create_path(best_match_key)
        if section_dir_path:
            print(f"[INFO] Using directory: '{section_dir_path}'")
        
        return (best_match_key, section_dir_path) if section_dir_path else (None, None)
    
    def parse_section(self, section_name: str, ticker: str, html_flag: bool = False) -> str:
        """Extract a specific section's text and save it."""

        if self.filing is None:
            print(
                f"[ERROR] Cannot parse section '{section_name}' for {ticker}: "
                "no filing object is attached to S1Parser (filing=None)."
            )
            return ""

        try:
            raw_content = IPOParser._get_raw_content(self.filing)
            content = extract_section_text(
                raw_content,
                self.toc or {},
                section_name,
                self.filing.filing_url.split(".")[-1],
            )
        except Exception as e:
            print(f"[ERROR] Failed to parse content for {ticker} - {section_name}: {e}")
            return ""

        if not content:
            print(f"[WARN] No content extracted for {ticker} - {section_name}.")
            return ""

        # MODIFICATION: Extract year logic (same as parse_toc)
        if hasattr(self.filing, 'filing_year') and self.filing.filing_year:
            year = str(self.filing.filing_year)
        elif hasattr(self.filing, 'filing_date') and self.filing.filing_date:
            year = self.filing.filing_date.split('-')[0]
        else:
            year = "unknown_year"

        # MODIFICATION: Pass year to the updated mapping function
        canonical_key, section_dir_path = self.normalize_and_map_section(
            section_name, 
            self.mappings_path, 
            self.output_base_path,
            year
        )

        if not canonical_key or not section_dir_path:
            print(f"[ERROR] Could not determine save path for {ticker} - {section_name}. Content not saved.")
            return content
        
        try:
            if html_flag:
                save_dir = os.path.join(section_dir_path, "html_files")
                filename = f"{ticker}_{canonical_key}.htm"
                file_path = os.path.join(save_dir, filename)
                
                os.makedirs(save_dir, exist_ok=True)
            else:
                filename = f"{ticker}_{canonical_key}.txt"
                file_path = os.path.join(section_dir_path, filename)

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
            print(f"[SUCCESS] Parsed and saved section to: {file_path}")
        
        except IOError as e:
            print(f"[ERROR] Failed to write file to {file_path}: {e}")
        except Exception as e:
            print(f"[ERROR] An unexpected error occurred during file save: {e}")

        return content
    
    def parse_company(self, ticker: str, html_flag: bool = False):
        """
        Parses all sections found in the Table of Contents for a given ticker,
        saving each section to its canonically-named directory.
        
        This method iterates through `self.toc.keys()` and calls 
        `self.parse_section()` for each one, reusing all existing
        parsing, mapping, and saving logic.
        
        Args:
            ticker: The company ticker (e.g., "SNOW")
        """

        if self.filing is None:
            print(
                f"[ERROR] Cannot parse company for ticker {ticker}: "
                "no filing object is attached to S1Parser (filing=None). "
                "This usually means the download step failed or returned None."
            )
            return

        if not self.toc:
            print(
                f"[ERROR] Table of Contents is empty for filing {self.filing.filing_url}. "
                "Cannot parse company."
            )
            return

        print(f"\n[INFO] === Starting full company parse for ticker: {ticker} ===")
        print(f"[INFO] Found {len(self.toc.keys())} sections in the Table of Contents.")
        
        section_count = 0
        for section_name in self.toc.keys():
            section_count += 1
            print(f"\n--- Parsing section {section_count}/{len(self.toc.keys())}: '{section_name}' ---")
            self.parse_section(section_name, ticker, html_flag=html_flag)
        
        print(f"\n[INFO] === Finished full company parse for {ticker} ===\n")
    
    @staticmethod
    def _get_raw_content(filing: Filing) -> str:
        """
        Get raw content from filing object or load from disk.
        
        Args:
            filing: S1Filing object
            
        Returns:
            str: The raw filing content (HTML or plain text)
        """
        if filing.raw_content:
            if filing.filing_url.endswith('.htm') or filing.filing_url.endswith('.html'): 
                return filing.raw_content
            # Handling ASCII multi-docs, select first doc
            DOC_PATTERN = re.compile(
                            r"""(?is)
                            <DOCUMENT\b[^>]*>
                            .*?                        
                            <TYPE>\s*S-1[^\r\n<]*
                            .*?                        
                            </DOCUMENT>                
                            """,
                            re.VERBOSE,
                        )
            match = DOC_PATTERN.search(filing.raw_content)
            return match.group(0)
        with open(f"../../../data/{filing.local_path}", 'r', encoding='utf-8') as f:
            return f.read()
        

############################################################
#################### EXAMPLE USAGE #########################
############################################################

# downloader = S1Downloader("slohani7@gatech.edu", "Georgia Institute of Technology")
# filing = downloader.download_s1(Company.from_ticker("SNOW"))

# parser = S1Parser(filing)
# risk_factor_text = parser.parse_section("Risk Factors")

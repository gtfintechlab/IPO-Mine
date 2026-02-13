"""
IPO Filing Parser

Provides high-level API for parsing IPO filings, including table of contents
extraction and section text extraction. Supports optional LLM-based validation.
"""
import json
import re
import os
from thefuzz import process
from ..entities import CompanyFilings, Filing
from .toc_parser import get_parser_for_year, parse_toc_plain_text
from .section_parser import extract_section_text_html, extract_section_text_ascii
from .llm_validator import LLMValidator, ValidationMode, validate_section_binary, validate_section_likert

from typing import Dict, Optional, Literal, Tuple, Any

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
    
    def normalize_and_map_section(self, input_text: str, mappings_path: str, output_base_path: str, year: str, verbose: bool = True):
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
            if verbose:
                print(f"[ERROR] Mappings file not found at: {mappings_path}")
            return None, None
        except json.JSONDecodeError:
            if verbose:
                print(f"[ERROR] Mappings file at {mappings_path} is not valid JSON.")
            return None, None

        cleaned_input = input_text.lower().strip()
        canonical_names = {key: data['canonical_name'] for key, data in sections.items()}

        def get_or_create_path(key):
            path = os.path.join(output_base_path, key, str(year))
            if not os.path.isdir(path):
                try:
                    os.makedirs(path)
                    if verbose:
                        print(f"[INFO] New section directory created: '{path}'")
                except OSError as e:
                    if verbose:
                        print(f"[ERROR] Could not create directory at {path}: {e}")
                    return None
            return path

        for key, name in canonical_names.items():
            if cleaned_input == name:
                if verbose:
                    print(f"[INFO] Input '{input_text}' is a direct match for canonical name: '{name}'.")
                section_dir_path = get_or_create_path(key)
                return (key, section_dir_path) if section_dir_path else (None, None)

        for key, data in sections.items():
            if cleaned_input in data['variants']:
                if verbose:
                    print(f"[INFO] Input '{input_text}' is already a known variant for '{data['canonical_name']}'.")
                section_dir_path = get_or_create_path(key)
                return (key, section_dir_path) if section_dir_path else (None, None)

        best_match_name, score = process.extractOne(cleaned_input, canonical_names.values())
        if not best_match_name:
            if verbose:
                print(f"[ERROR] Could not find any canonical names to match against.")
            return None, None
            
        best_match_key = next(key for key, name in canonical_names.items() if name == best_match_name)

        if verbose:
            print(f"[ACTION] New variant found. Fuzzy matched '{input_text}' to '{best_match_name}' with score {score}.")
        sections[best_match_key]['variants'].append(cleaned_input)
        
        try:
            with open(mappings_path, 'w', encoding='utf-8') as f:
                json.dump(sections, f, indent=2)
            if verbose:
                print(f"[ACTION] Added '{cleaned_input}' as a new variant and updated '{mappings_path}'.")
        except IOError as e:
            if verbose:
                print(f"[ERROR] Could not write to mappings file at {mappings_path}: {e}")
            return None, None

        section_dir_path = get_or_create_path(best_match_key)
        if section_dir_path:
            if verbose:
                print(f"[INFO] Using directory: '{section_dir_path}'")
        
        return (best_match_key, section_dir_path) if section_dir_path else (None, None)

    def parse_section(
        self, 
        section_name: str, 
        save: bool = False, 
        plaintext: bool = True, 
        verbose: bool = False,
        validate: bool = False,
        validation_provider: str = "anthropic",
        validation_mode: str = "binary",
        validation_api_key: Optional[str] = None,
    ) -> Tuple[str, Optional[str], Optional[Dict[str, Any]]]:
        """
        Extract a specific section's text, optionally validate it with LLM, and save it.
        
        Args:
            section_name: Name of the section to parse
            save: Whether to save the parsed content to disk
            plaintext: Whether to return plaintext (vs HTML)
            verbose: Whether to print progress information
            validate: Whether to validate parsed content with LLM
            validation_provider: LLM provider ("anthropic", "google", "openai", "huggingface")
            validation_mode: Validation mode ("binary" or "likert")
            validation_api_key: Optional API key for validation (uses environment variable if not provided)
            
        Returns:
            Tuple of (content, filename, validation_result)
            - content: Extracted section text
            - filename: Path where file was saved (None if not saved)
            - validation_result: Validation result dict (None if not validated)
        """

        if self.filing is None:
            if verbose:
                print(f"[ERROR] Cannot parse section '{section_name}': no filing object is attached to IPOParser (filing=None).")
            return "", None, None

        try:
            raw_content = IPOParser._get_raw_content(self.filing)
            file_extension = self.filing.filing_url.split(".")[-1]
            if file_extension == "txt":
                content = extract_section_text_ascii(raw_content, self.toc or {}, section_name, file_extension, verbose=verbose)
            else:
                content = extract_section_text_html(raw_content, self.toc or {}, section_name, file_extension, not plaintext, verbose=verbose)

        except Exception as e:
            if verbose:
                import traceback
                print(f"[ERROR] Failed to parse content for {section_name}: {e}")
                print(traceback.format_exc())
            return "", None, None

        if not content:
            if verbose:
                print(f"[WARN] No content extracted for {section_name}.")
            return "", None, None

        filename = None
        if save:
            filename = self._save_section_content(content, section_name, not plaintext, verbose)

        # Validate content with LLM if requested
        validation_result = None
        if validate:
            try:
                if verbose:
                    print(f"[INFO] Validating section '{section_name}' with {validation_provider} ({validation_mode} mode)...")
                
                if validation_mode == "binary":
                    result = validate_section_binary(
                        content, 
                        section_name,
                        provider=validation_provider,
                        api_key=validation_api_key
                    )
                elif validation_mode == "likert":
                    result = validate_section_likert(
                        content,
                        section_name,
                        provider=validation_provider,
                        api_key=validation_api_key
                    )
                else:
                    if verbose:
                        print(f"[ERROR] Unknown validation mode: {validation_mode}")
                    return content, filename, None
                
                validation_result = {
                    "provider": validation_provider,
                    "mode": validation_mode,
                    "answer": result.answer,
                    "justification": result.justification,
                }
                
                if verbose:
                    print(f"[SUCCESS] Validation complete. Answer: {result.answer}")
                    
            except Exception as e:
                if verbose:
                    import traceback
                    print(f"[ERROR] Validation failed for section '{section_name}': {e}")
                    print(traceback.format_exc())
                # Don't fail parsing if validation fails
                validation_result = {"error": str(e)}

        return content, filename, validation_result
    
    def parse_company(
        self, 
        ticker: str, 
        html_flag: bool = False,
        validate: bool = False,
        validation_provider: str = "anthropic",
        validation_mode: str = "binary",
        validation_api_key: Optional[str] = None,
    ):
        """
        Parses all sections found in the Table of Contents for a given ticker,
        saving each section to its canonically-named directory.
        
        This method iterates through `self.toc.keys()` and calls 
        `self.parse_section()` for each one, reusing all existing
        parsing, mapping, and saving logic.
        
        Args:
            ticker: The company ticker (e.g., "SNOW")
            html_flag: Whether to save as HTML format
            validate: Whether to validate each section with LLM
            validation_provider: LLM provider for validation
            validation_mode: Validation mode ("binary" or "likert")
            validation_api_key: Optional API key for validation
            
        Returns:
            Dictionary with parsing and validation results for all sections
        """

        if self.filing is None:
            print(
                f"[ERROR] Cannot parse company for ticker {ticker}: "
                "no filing object is attached to S1Parser (filing=None). "
                "This usually means the download step failed or returned None."
            )
            return {}

        if not self.toc:
            print(
                f"[ERROR] Table of Contents is empty for filing {self.filing.filing_url}. "
                "Cannot parse company."
            )
            return {}

        print(f"\n[INFO] === Starting full company parse for ticker: {ticker} ===")
        print(f"[INFO] Found {len(self.toc.keys())} sections in the Table of Contents.")
        
        results = {}
        section_count = 0
        for section_name in self.toc.keys():
            section_count += 1
            print(f"\n--- Parsing section {section_count}/{len(self.toc.keys())}: '{section_name}' ---")
            content, filename, validation_result = self.parse_section(
                section_name, 
                save=True,
                verbose=True,
                validate=validate,
                validation_provider=validation_provider,
                validation_mode=validation_mode,
                validation_api_key=validation_api_key,
            )
            results[section_name] = {
                "filename": filename,
                "content_length": len(content),
                "validation": validation_result,
            }
        
        print(f"\n[INFO] === Finished full company parse for {ticker} ===\n")
        return results
    
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
                            <TYPE>\s*(?:S-1|F-1)[^\r\n<]*
                            .*?
                            </DOCUMENT>
                            """,
                            re.VERBOSE,
                        )
            match = DOC_PATTERN.search(filing.raw_content)
            return match.group(0)
        if os.path.isabs(filing.local_path):
            file_path = filing.local_path
        else:
             # filing.local_path is relative to data/ (e.g. "raw/filings/...") or relative to raw/?
             # Downloader sets it as: relative_filename = filename.resolve().relative_to(self.download_root.parent.resolve())
             # If download_root is data/raw, parent is data/. So local_path starts with "raw/filings/..."
             # We want to join it with DATA_DIR (which should be the same as download_root.parent)
             # But let's check config.
             from ..utils.config import DATA_DIR
             file_path = os.path.join(DATA_DIR, filing.local_path)
             
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
        
    def _save_section_content(self, content: str, section_name: str, is_html: bool, verbose: bool = False) -> Optional[str]:
        """Save section content to disk.
        
        Args:
            content: The content to save
            section_name: Name of the section
            is_html: Whether content is HTML format (determines file extension)
            verbose: If True, print progress information
            
        Returns:
            Path where file was saved, or None if save failed
        """
        year = self.filing.filing_date.split('-')[0]
        canonical_key, section_dir_path = self.normalize_and_map_section(
            section_name, 
            self.mappings_path, 
            self.output_base_path,
            year,
            verbose=verbose
        )
        
        if not canonical_key or not section_dir_path:
            if verbose:
                print(f"[ERROR] Could not determine save path for section '{section_name}'.")
            return None

        try:

            # TODO: only doing this for now (testing), remove later
            # Slug generation
            company_name_for_slug = "Unknown"
            if hasattr(self.filing, 'name') and self.filing.name:
                company_name_for_slug = self.filing.name
            
            slug = self._create_company_slug(company_name_for_slug)
            
            # Filing object has 'cik' but not 'tickers'
            # We can use CIK to identify
            if hasattr(self.filing, 'cik') and self.filing.cik:
                 identifier = f'CIK{self.filing.cik}_{slug}'
            else:
                 # Fallback if even CIK is missing (unlikely for valid Filing)
                 identifier = f'FILE_{slug}'


            extension = '.htm' if is_html else '.txt'
            filename = f"{identifier}_{canonical_key}{extension}"
            
            save_dir = os.path.join(section_dir_path, "html_files") if is_html else section_dir_path
            file_path = os.path.join(save_dir, filename)
            
            os.makedirs(save_dir, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            if verbose:
                print(f"[SUCCESS] Saved section to: {file_path}")
            
            return file_path
        
        except Exception as e:
            import traceback
            if verbose:
                print(f"[ERROR] Failed to save section '{section_name}': {e}")
                print(traceback.format_exc())
            return None
        
    def _create_company_slug(self, text: str) -> str:
        slug = re.sub(r'[^a-zA-Z0-9]+', '-', text.lower()).strip('-')
        if len(slug) > 10:
            slug = slug[:10].rsplit('-', 1)[0]
        return slug

############################################################
#################### EXAMPLE USAGE #########################
############################################################

# Basic parsing without validation
# parser = IPOParser(filing, mappings_path, output_base_path)
# content, filename, _ = parser.parse_section("Risk Factors", save=True)

# Parse with binary validation (Yes/No)
# content, filename, validation = parser.parse_section(
#     "Risk Factors",
#     save=True,
#     validate=True,
#     validation_provider="anthropic",  # or "google", "openai", "huggingface"
#     validation_mode="binary"
# )
# print(f"Validation: {validation['answer']}")
# print(f"Justification: {validation['justification']}")

# Parse with Likert validation (1-5 confidence scale)
# content, filename, validation = parser.parse_section(
#     "Risk Factors",
#     save=True,
#     validate=True,
#     validation_provider="huggingface",  # Use open source
#     validation_mode="likert"
# )
# print(f"Confidence: {validation['answer']}/5")

# Parse entire company with validation
# results = parser.parse_company(
#     "SNOW",
#     validate=True,
#     validation_provider="google",  # Fastest/cheapest
#     validation_mode="binary"
# )
# for section, result in results.items():
#     print(f"{section}: {result['validation']}")

"""
Section extraction utilities for S1 filings.

This module provides functions to parse S1 filings into pages and extract
specific sections based on a table of contents.
"""

from typing import Dict, Tuple, Optional, List
import re
from bs4 import BeautifulSoup

try:
    from thefuzz import fuzz

    HAS_FUZZY = True
except ImportError:
    HAS_FUZZY = False

# Try to use lxml parser for BeautifulSoup (5-10x faster than html.parser)
try:
    import lxml
    BS_PARSER = "lxml"
except ImportError:
    BS_PARSER = "html.parser"

# Pre-compiled regex patterns for fast HTML text extraction
_HTML_TAG_RE = re.compile(r'<[^>]+>')
_HTML_ENTITY_RE = re.compile(r'&(?:#\d+|#x[0-9a-fA-F]+|[a-zA-Z]+);')
_MULTI_WHITESPACE_RE = re.compile(r'[ \t]+')
_MULTI_NEWLINE_RE = re.compile(r'\n{3,}')

# Common HTML entities mapping for fast replacement
_HTML_ENTITIES = {
    '&nbsp;': ' ', '&amp;': '&', '&lt;': '<', '&gt;': '>',
    '&quot;': '"', '&apos;': "'", '&#160;': ' ', '&#39;': "'",
    '&mdash;': '—', '&ndash;': '–', '&hellip;': '…',
    '&ldquo;': '"', '&rdquo;': '"', '&lsquo;': ''', '&rsquo;': ''',
}


def _fast_html_to_text(html: str) -> str:
    """
    Fast HTML to text conversion using regex.
    
    This is 10-50x faster than BeautifulSoup for simple text extraction.
    Use this when you don't need sophisticated HTML handling.
    
    Args:
        html: HTML string to convert
        
    Returns:
        Plain text with HTML tags and entities removed
    """
    if not html:
        return ""
    
    # Replace common block elements with newlines for structure
    text = re.sub(r'<(?:br|BR)\s*/?>', '\n', html)
    text = re.sub(r'</(?:p|P|div|DIV|tr|TR|li|LI|h[1-6]|H[1-6])>', '\n', text)
    
    # Remove all HTML tags
    text = _HTML_TAG_RE.sub('', text)
    
    # Replace common HTML entities
    for entity, char in _HTML_ENTITIES.items():
        text = text.replace(entity, char)
    
    # Remove remaining HTML entities (numeric and named)
    text = _HTML_ENTITY_RE.sub(' ', text)
    
    # Normalize whitespace
    text = _MULTI_WHITESPACE_RE.sub(' ', text)
    text = _MULTI_NEWLINE_RE.sub('\n\n', text)
    
    return text.strip()


def _get_text_from_html(html: str, use_fast: bool = True) -> str:
    """
    Extract text from HTML with automatic method selection.
    
    Args:
        html: HTML string
        use_fast: If True, use fast regex method; if False, use BeautifulSoup
        
    Returns:
        Plain text extracted from HTML
    """
    if use_fast:
        return _fast_html_to_text(html)
    else:
        soup = BeautifulSoup(html, BS_PARSER)
        text = soup.get_text("\n")
        return _MULTI_NEWLINE_RE.sub('\n\n', text).strip('\n')


# Module-level constants
ROMAN_RE = re.compile(r"^[ivxlcdm]+$", re.IGNORECASE)
ROMAN_MAP = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
ALPHA_PAGE_RE = re.compile(r"^([A-Z]+)\s*[-\s]?\s*([0-9]+)$", re.IGNORECASE)
STOPWORDS = {"of", "and", "the", "to", "in", "for", "on", "with", "at", "by", "from"}

# Words that can be optionally present in section headers
OPTIONAL_WORDS = {"the", "of", "and", "to"}

# Known major section names (in normalized form) that indicate a top-level section
# These should NOT be treated as subsections of Risk Factors, etc.
MAJOR_SECTION_NAMES = {
    "use of proceeds", "dilution", "capitalization", "dividend policy",
    "selected financial data", "selected consolidated financial data",
    "management's discussion and analysis", "business", "management",
    "principal stockholders", "principal shareholders", "certain transactions",
    "description of capital stock", "description of securities",
    "shares eligible for future sale", "underwriting", "legal matters",
    "experts", "additional information", "financial statements",
    "index to financial statements", "prospectus summary", "summary",
    "the company", "our company", "the offering", "plan of distribution",
    "selling stockholders", "selling shareholders", "executive compensation",
    "forward-looking statements", "cautionary statement", "where you can find more information",
}


def _build_flexible_section_pattern(section_name: str, sep: str = r"[ \t\r\n]+", require_word_boundaries: bool = False) -> str:
    """
    Build a regex pattern that matches a section name with optional articles/prepositions.
    
    This handles cases where:
    - TOC says "Price Range of the Common Stock" but header is "PRICE RANGE OF COMMON STOCK"
    - TOC says "The Company" but header is "COMPANY"
    - TOC has "PROSPECTUS The Company" (malformed) but header is "THE COMPANY"
    
    Args:
        section_name: The section name from TOC
        sep: Whitespace separator pattern (default: any whitespace)
        require_word_boundaries: If True, require the pattern to match at word boundaries
            (prevents "Exchange Offer" from matching within "EXCHANGE OFFER PROCEDURES")
        
    Returns:
        A regex pattern string that flexibly matches the section name
    """
    # Clean up malformed section names (e.g., "PROSPECTUS The Company" -> "The Company")
    # Remove common prefixes that might be erroneously included
    prefixes_to_strip = ['prospectus', 'summary', 'exhibit']
    cleaned_name = section_name.strip()
    lower_name = cleaned_name.lower()
    for prefix in prefixes_to_strip:
        if lower_name.startswith(prefix + ' ') and len(lower_name) > len(prefix) + 5:
            # Only strip if there's substantial text after the prefix
            remaining = cleaned_name[len(prefix):].strip()
            if remaining and remaining[0].isupper():
                cleaned_name = remaining
                break
    
    # Split into words
    words = cleaned_name.split()
    
    # Get required words (non-optional)
    required_words = [w for w in words if w.lower() not in OPTIONAL_WORDS]
    
    # If we'd end up with fewer than 2 required words, keep all words as required
    # This prevents patterns like "Company" from matching everywhere
    if len(required_words) < 2:
        # Use all words as required, but still case-insensitive
        pattern = '(?i)' + sep.join(re.escape(w) for w in words)
    else:
        # Build pattern: required words separated by flexible whitespace
        # with optional words allowed in between
        optional_group = r'(?:(?:the|of|and|to)' + sep + r')*'
        pattern = '(?i)' + (sep + optional_group).join(re.escape(w) for w in required_words)
    
    # Add word boundary check if requested
    if require_word_boundaries:
        # After the last word, ensure we're at a real boundary (not part of longer phrase)
        # 
        # Key insight: Header continuations are ALL CAPS (e.g., "EXCHANGE OFFER PROCEDURES")
        # while content starts with a capital then continues lowercase ("In addition to...")
        #
        # So we block: whitespace + ALL CAPS word (2+ uppercase letters)
        # But allow: whitespace + Title Case word (uppercase letter followed by lowercase)
        #
        # IMPORTANT: Use (?-i:...) to make the uppercase check case-SENSITIVE
        # even though the rest of the pattern is case-insensitive
        # This ensures [A-Z] only matches actual uppercase letters
        pattern = pattern + r'(?!\s+(?-i:[A-Z]{2,}))'
    
    return pattern


def _is_likely_subsection(toc_entry: str, parent_section: str, toc: Dict[str, str]) -> bool:
    """
    Determine if a TOC entry is likely a subsection of the parent section.
    
    This handles cases where Risk Factors subsections are listed in the TOC:
    - "We May Fail to Obtain More Profitable Service Contracts..."
    - "Risks Associated with Our Business"
    - "Our Principal Stockholder Controls Us..."
    
    Args:
        toc_entry: The candidate TOC entry to check
        parent_section: The parent section name (e.g., "Risk Factors")
        toc: Full TOC dictionary for page context
    
    Returns:
        True if likely a subsection, False if likely a major section
    """
    entry_lower = toc_entry.lower().strip()
    parent_lower = parent_section.lower().strip()
    
    # Check if it's a known major section (definitely not a subsection)
    entry_normalized = _normalize_section_key(toc_entry)
    for major in MAJOR_SECTION_NAMES:
        major_normalized = re.sub(r"[^a-z0-9]+", "", major)
        if entry_normalized == major_normalized or entry_lower == major:
            return False
    
    # If it's the same section, not a subsection
    if entry_normalized == _normalize_section_key(parent_section):
        return False
    
    # Check page numbers - subsections often have same or close page number
    parent_page = toc.get(parent_section, "")
    entry_page = toc.get(toc_entry, "")
    try:
        parent_page_num = int(parent_page) if parent_page.isdigit() else 0
        entry_page_num = int(entry_page) if entry_page.isdigit() else 0
        # Subsections are typically within 10 pages of parent start (expanded from 5)
        # Risk Factors can span many pages
        same_region = abs(entry_page_num - parent_page_num) <= 10
    except (ValueError, AttributeError):
        same_region = False
    
    # Risk Factors subsection heuristics
    if "risk" in parent_lower:
        # Subsection patterns for Risk Factors:
        # - Starts with "We ", "Our ", "The ", "If ", "Risks ", "There ", "A ", "An "
        # - Contains "may", "could", "might", "risk", "uncertain"
        # - Is a long descriptive sentence (>5 words)
        subsection_starters = (
            "we ", "our ", "the ", "if ", "risks ", "certain ", "limited ",
            "there ", "a ", "an ", "difficulties ", "additional ", "changes ",
            "dependence ", "failure ", "loss ", "absence ", "potential ",
        )
        has_subsection_starter = entry_lower.startswith(subsection_starters)
        
        risk_keywords = (
            "may ", "could ", "might ", "risk", "uncertain", "depend",
            "adversely", "affect", "harm", "damage", "loss", "fail",
            "limit", "restrict", "prevent", "subject", "expose",
        )
        has_risk_keyword = any(kw in entry_lower for kw in risk_keywords)
        
        is_descriptive_sentence = len(toc_entry.split()) >= 5
        
        # Strong indicator: starts with subsection pattern AND has risk keywords
        if has_subsection_starter and has_risk_keyword:
            return True
        
        # Strong indicator: has risk keyword and is on same page or within region
        if has_risk_keyword and same_region:
            return True
        
        # Moderate indicator: descriptive sentence on same page region
        if is_descriptive_sentence and same_region:
            # Additional check: does it look like a sentence vs a section title?
            # Section titles are usually 1-4 words, capitalized or ALL CAPS
            # Subsection descriptions are 5+ words, often sentence-case
            if not toc_entry.isupper():
                return True
    
    return False


def _find_next_major_section(toc: Dict[str, str], current_section: str, current_idx: int) -> Optional[str]:
    """
    Find the next major section after the current section, skipping subsections.
    
    Args:
        toc: Table of contents dictionary
        current_section: Current section name
        current_idx: Index of current section in TOC
    
    Returns:
        Name of next major section, or None if not found
    """
    toc_keys = list(toc.keys())
    current_normalized = _normalize_section_key(current_section)
    
    for i in range(current_idx + 1, len(toc_keys)):
        candidate = toc_keys[i]
        candidate_normalized = _normalize_section_key(candidate)
        
        # Skip duplicates of the same section (e.g., "Risk Factors" at page 3 and "RISK FACTORS" at page 17)
        # These are the same section, not a boundary
        if candidate_normalized == current_normalized:
            continue
        
        if not _is_likely_subsection(candidate, current_section, toc):
            return candidate
    
    return None


def _normalize_section_key(text: str) -> str:
    if not text:
        return ""
    text = (
        text.replace("\xa0", " ")
        .replace("\u200b", "")
        .replace("\u200c", "")
        .replace("\u200d", "")
        .replace("\u2019", "'")
        .replace("\u2018", "'")
    )
    text = text.lower()
    return re.sub(r"[^a-z0-9]+", "", text)


def _looks_like_toc_entry_line(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return False
    lower = s.lower()
    if "table of contents" in lower:
        return True
    if re.search(r"\.{2,}\s*\(?[ivxlcdm]{1,8}\)?\s*$", s, re.IGNORECASE):
        return True
    if re.search(r"\.{2,}\s*\d{1,4}\s*$", s):
        return True
    if re.search(r"\.{2,}\s*[A-Z]\s*-\s*\d{1,3}\s*$", s, re.IGNORECASE):
        return True
    if re.search(r"\s+\(?[ivxlcdm]{1,8}\)?\s*$", s, re.IGNORECASE):
        return True
    if re.search(r"\s+\d{1,4}\s*$", s):
        return True
    if re.search(r"\s+[A-Z]\s*-\s*\d{1,3}\s*$", s, re.IGNORECASE):
        return True
    return False


def _is_likely_real_section_header(lines: List[str], header_idx: int, section_name: str) -> bool:
    """
    Check if a header match is likely the real section header vs a summary reference.
    
    Real section headers are typically followed by:
    - Multiple paragraphs of content
    - Subsection headers (e.g., "RECENT LOSSES", "DEPENDENCE ON KEY PERSONNEL")
    - Warning language in all caps
    
    Summary references are typically followed by:
    - Brief summary paragraph (1-3 sentences)
    - Another section header (like "SUMMARY FINANCIAL INFORMATION")
    """
    if header_idx >= len(lines) - 5:
        return True  # Near end of document, take what we can get
    
    # Look at the next 30 lines to evaluate content
    following_lines = lines[header_idx + 1:header_idx + 35]
    following_text = '\n'.join(following_lines)
    
    # Check for signs of a real Risk Factors section:
    # 1. All-caps warning language typical of Risk Factors
    risk_indicators = [
        'SUBSTANTIAL INVESTMENT RISKS',
        'HIGH DEGREE OF RISK',
        'AFFORD THE LOSS',
        'LOSS OF THEIR ENTIRE INVESTMENT',
        'CAREFULLY CONSIDER',
        'SHOULD CAREFULLY',
    ]
    has_risk_warning = any(ind in following_text.upper() for ind in risk_indicators)
    
    # 2. Multiple subsection headers (all-caps lines that are subsection titles)
    subsection_count = 0
    for line in following_lines:
        stripped = line.strip()
        if stripped and stripped.isupper() and len(stripped) > 5 and len(stripped) < 60:
            # Looks like a subsection header
            if not any(skip in stripped for skip in ['TABLE', 'PAGE', '---', '===']):
                subsection_count += 1
    
    # 3. Check for premature end (another major section starting soon)
    next_section_markers = [
        'SUMMARY FINANCIAL INFORMATION',
        'USE OF PROCEEDS',
        'SELECTED FINANCIAL DATA',
        'CAPITALIZATION',
        'DILUTION',
    ]
    has_early_next_section = any(
        marker in following_text.upper()[:800]  # Within first ~15 lines
        for marker in next_section_markers
    )
    
    # Real section: has risk warning OR multiple subsections, without early next section
    if has_risk_warning and not has_early_next_section:
        return True
    if subsection_count >= 2 and not has_early_next_section:
        return True
    if has_early_next_section and not has_risk_warning:
        return False  # Summary reference followed by next section
        
    return True  # Default to accepting if unsure


def _find_header_line_index(lines: List[str], section_name: str) -> Optional[int]:
    target = _normalize_section_key(section_name)
    if not target:
        return None
    
    # Collect all potential matches first
    potential_matches = []
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if not line_stripped:
            continue
        if _looks_like_toc_entry_line(line_stripped):
            continue
        normalized_line = _normalize_section_key(line_stripped)
        if not normalized_line:
            continue
        
        # Check if this looks like a real section header
        if (
            normalized_line == target
            or normalized_line.startswith(target)
            or normalized_line.endswith(target)
        ):
            # Reject lines that look like cross-references or subsection references
            # e.g., "Risk Factors - Non-Registration..." is NOT a header
            # e.g., '"RISK FACTORS" BEGINNING AT PAGE 10' is NOT a header
            # Real headers: "RISK FACTORS" or "RISK FACTORS:" or "RISK FACTORS\n"
            
            upper_stripped = line_stripped.upper()
            section_upper = section_name.upper()
            
            # Find where the section name ends in the line
            section_pos = upper_stripped.find(section_upper)
            if section_pos != -1:
                after_section = line_stripped[section_pos + len(section_name):].strip()
                
                # Check for various cross-reference patterns:
                
                # 1. Dash/hyphen followed by more content (e.g., "Risk Factors - Specific Topic")
                if after_section and after_section[0] in '-–—':
                    after_dash = after_section[1:].strip()
                    if len(after_dash) > 3:
                        continue  # Skip this match
                
                # 2. Quoted section name followed by page reference
                # e.g., '"RISK FACTORS" BEGINNING AT PAGE' or '"RISK FACTORS" (PAGE'
                after_upper = after_section.upper()
                page_ref_patterns = [
                    'BEGINNING AT PAGE',
                    'BEGINNING ON PAGE', 
                    'AT PAGE',
                    'ON PAGE',
                    '(PAGE',
                    '" (PAGE',
                    'SEE PAGE',
                ]
                if any(pat in after_upper for pat in page_ref_patterns):
                    continue  # Skip - this is a page reference
                
                # 3. Check if preceded by "SEE" (common cross-reference pattern)
                before_section = line_stripped[:section_pos].strip().upper()
                if before_section.endswith('SEE') or before_section.endswith('SEE "'):
                    continue  # Skip - this is a "See Risk Factors" reference
                
                # 4. Check if preceded by "heading" patterns (cross-reference like 'under the heading "Risk Factors"')
                cross_ref_before_patterns = [
                    'THE HEADING "', 'HEADING "', 'UNDER "', 'CAPTIONED "', 
                    'TITLED "', 'ENTITLED "', 'UNDER THE "', 'IN THE "',
                    'REFER TO "', 'REFERRED TO AS "', 'SET FORTH IN "',
                    'CONTAINED IN "', 'DESCRIBED IN "', 'DISCUSSED IN "',
                    'READ THE "', 'READ "', 'THE "'
                ]
                if any(before_section.endswith(pat) for pat in cross_ref_before_patterns):
                    continue  # Skip - this is a heading reference
                
                # 5. Check if quoted section name followed by comma or "and" (mid-sentence reference)
                # e.g., '"Risk Factors," among others' or '"Risk Factors" and elsewhere'
                after_patterns = [
                    '," ', # Quoted followed by comma and space (part of list)
                    '" AND ', # Quoted followed by "and"
                    '" OR ', # Quoted followed by "or" 
                    '" THAT ', # "Risk Factors" that could cause...
                    '" AMONG ', # "Risk Factors," among others
                    '" COULD ', # "Risk Factors," could cause
                    '" MAY ', # "Risk Factors" may
                    '" SECTION', # the "Risk Factors" section
                    '" CONTAINED', # "Risk Factors" contained
                    '" INCLUDED', # "Risk Factors" included
                    '" ELSEWHERE', # "Risk Factors" elsewhere
                ]
                if any(after_upper.startswith(pat) for pat in after_patterns):
                    continue  # Skip - this is a mid-sentence reference
                
                # 6. Check if line has too many words (real headers are usually short)
                # Real headers: "RISK FACTORS" or "RISK FACTORS:" (2-5 words typically)
                # Cross-references: "Certain statements contained in... Risk Factors..." (many words)
                line_words = line_stripped.split()
                if len(line_words) > 8 and section_pos > 10:
                    # Line is long and section name is not near the start - likely cross-reference
                    continue
            
            potential_matches.append(i)
    
    # If we found multiple matches, use heuristics to pick the best one
    if not potential_matches:
        return None
    
    if len(potential_matches) == 1:
        return potential_matches[0]
    
    # For Risk Factors specifically, prefer the one that looks like the real section
    if 'risk' in section_name.lower():
        for idx in potential_matches:
            if _is_likely_real_section_header(lines, idx, section_name):
                return idx
    
    # Default to first match
    return potential_matches[0]


def _build_loose_section_pattern(section_name: str, allow_tags: bool = False) -> str:
    name = (section_name or "").strip()
    if not name:
        return ""
    name = name.replace("\u2019", "'").replace("\u2018", "'")
    words = [w for w in re.split(r"\s+", name) if w]
    token_patterns = []
    for word in words:
        token = re.escape(word)
        token = token.replace("\\'s", "(?:'s|\u2019s|s)")
        token = token.replace("\\&", "(?:&|and)")
        token_patterns.append(token)
    separator = r"(?:\s|&nbsp;|&#160;|<[^>]+>)*" if allow_tags else r"\s*"
    return separator.join(token_patterns)


def _normalize_page_label(label: str) -> str:
    if not label:
        return label
    text = label.strip()
    text = text.translate(
        str.maketrans(
            {
                "\u2010": "-",
                "\u2011": "-",
                "\u2012": "-",
                "\u2013": "-",
                "\u2014": "-",
                "\u2015": "-",
                "\u2212": "-",
            }
        )
    )
    text = text.strip("-")
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1].strip()
    text = re.sub(r"^page\s+", "", text, flags=re.IGNORECASE)
    return text.strip()


def roman_to_int(s: str) -> int:
    """Convert Roman numeral to integer."""
    s = s.upper()
    total = 0
    for i, ch in enumerate(s):
        val = ROMAN_MAP[ch]
        if i + 1 < len(s) and ROMAN_MAP[s[i + 1]] > val:
            total -= val
        else:
            total += val
    return total


def page_to_int(p: str) -> int:
    """Convert page string (Roman or Arabic) to integer."""
    p = _normalize_page_label(p)
    if not p:
        raise ValueError("Empty page label")

    if ROMAN_RE.match(p):
        return roman_to_int(p)

    if p.isdigit():
        return int(p)

    match = ALPHA_PAGE_RE.match(p)
    if match:
        prefix, number = match.groups()
        prefix = prefix.upper()
        offset = 0
        for ch in prefix:
            offset = offset * 26 + (ord(ch) - ord("A") + 1)
        return offset * 1000 + int(number)

    cleaned = p.replace(",", "")
    if cleaned.isdigit():
        return int(cleaned)

    raise ValueError(f"Unsupported page label: '{p}'")


def _fuzzy_match_section(
    section_name: str, available_sections: List[str], threshold: int = 80
) -> Optional[str]:
    """
    Find a section name using fuzzy matching as a fallback.

    Args:
        section_name: The section name to search for
        available_sections: List of available section names
        threshold: Fuzzy match score threshold (0-100)

    Returns:
        Matched section name or None
    """
    if not HAS_FUZZY:
        return None

    best_match = None
    best_score = threshold

    for available in available_sections:
        score = fuzz.token_set_ratio(section_name.lower(), available.lower())
        if score > best_score:
            best_score = score
            best_match = available

    return best_match


def get_section_page_range(
    toc: Dict[str, str], section_name: str, verbose: bool = False
) -> Tuple[int, int]:
    """
    Get the start and end page numbers for a section from the ToC.

    Args:
        toc: Table of contents dictionary {section_name: page_number}
        section_name: Name of the section to find
        verbose: Print debug messages

    Returns:
        Tuple of (start_page, end_page) as integers

    Raises:
        KeyError: If section not found in ToC
    """
    keys = list(toc.keys())
    idx = None

    normalized_section = _normalize_section_key(section_name)

    # Try exact match first (normalized)
    for i, k in enumerate(keys):
        if _normalize_section_key(k) == normalized_section:
            idx = i
            break

    # Fallback 1: Try fuzzy matching
    if idx is None and HAS_FUZZY:
        matched_section = _fuzzy_match_section(section_name, keys, threshold=75)
        if matched_section:
            if verbose:
                print(f"[FUZZY MATCH] '{section_name}' matched to '{matched_section}'")
            idx = keys.index(matched_section)

    # Fallback 2: Try normalized containment match
    if idx is None and normalized_section:
        for i, k in enumerate(keys):
            normalized_key = _normalize_section_key(k)
            if (
                normalized_section in normalized_key
                or normalized_key in normalized_section
            ):
                if verbose:
                    print(f"[PARTIAL MATCH] '{section_name}' matched to '{k}'")
                idx = i
                break

    # Fallback 3: Risk Factors variants
    if idx is None and normalized_section == "riskfactors":
        for i, k in enumerate(keys):
            normalized_key = _normalize_section_key(k)
            if "risk" in normalized_key:
                if verbose:
                    print(f"[RISK MATCH] '{section_name}' matched to '{k}'")
                idx = i
                break

    if idx is None:
        raise KeyError(
            f"Section '{section_name}' not found in the Table of Contents. "
            f"Available sections are: {list(toc.keys())}"
        )

    start_page = page_to_int(toc[keys[idx]])
    end_page = (
        page_to_int(toc[keys[idx + 1]]) if idx + 1 < len(keys) else (start_page + 100)
    )

    return start_page, end_page


def extract_main_document(raw_text: str) -> str:
    """
    Extract only the main document (SEQUENCE 1) from a filing, excluding exhibits.
    
    SEC filings contain multiple documents:
    - Main prospectus (SEQUENCE 1, TYPE like S-1, S-1/A, etc.)
    - Exhibits (SEQUENCE > 1, TYPE like EX-10.1, EX-21, etc.)
    
    Exhibits often have their own page numbering that conflicts with the main document.
    This function extracts only the main document to avoid page number collisions.
    
    Args:
        raw_text: The complete filing text
        
    Returns:
        The main document content only (or original text if structure not found)
    """
    # Split by <DOCUMENT> tags
    doc_pattern = re.compile(r'<DOCUMENT>(.*?)</DOCUMENT>', re.DOTALL | re.IGNORECASE)
    documents = doc_pattern.findall(raw_text)
    
    if not documents:
        # No document structure found, return original text
        return raw_text
    
    # Find the main document (SEQUENCE 1 or non-exhibit type)
    for doc in documents:
        # Check if this is SEQUENCE 1 (main document)
        seq_match = re.search(r'<SEQUENCE>\s*1\s*(?:\n|<)', doc, re.IGNORECASE)
        if seq_match:
            # Verify it's not an exhibit type
            type_match = re.search(r'<TYPE>\s*([^\n<]+)', doc, re.IGNORECASE)
            if type_match:
                doc_type = type_match.group(1).strip().upper()
                # Main prospectus types: S-1, S-1/A, 424B1, etc.
                # Exhibit types: EX-*, GRAPHIC, XML, etc.
                if not doc_type.startswith('EX-') and doc_type not in ('GRAPHIC', 'XML', 'ZIP', 'COVER'):
                    return doc
    
    # Fallback: find the first non-exhibit document
    for doc in documents:
        type_match = re.search(r'<TYPE>\s*([^\n<]+)', doc, re.IGNORECASE)
        if type_match:
            doc_type = type_match.group(1).strip().upper()
            if not doc_type.startswith('EX-') and doc_type not in ('GRAPHIC', 'XML', 'ZIP', 'COVER'):
                return doc
    
    # Last resort: return the first document
    return documents[0] if documents else raw_text


def pages_by_bottom_number(raw_text: str) -> Dict[str, str]:
    """Parse plain text filing into pages based on page numbers at bottom."""
    # Extract only the main document (exclude exhibits which have their own page numbers)
    main_doc = extract_main_document(raw_text)
    text = main_doc.replace("\r\n", "\n").replace("\r", "\n")
    tag_re = re.compile(r"(?im)(?:^\s*<PAGE[^>]*>\s*(\d+)?\s*$)|(?:</page>)")
    num_re = re.compile(r"^\s*-?(\d{1,4})-?\s*$")
    roman_re = re.compile(r"^\s*-?\(?([IVXLCDMivxlcdm]{1,10})\)?-?\s*$")
    alpha_re = re.compile(r"^\s*([A-Z]\s*-\s*\d+)\s*$", re.IGNORECASE)

    matches = list(tag_re.finditer(text))
    spans = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        spans.append((m.group(1), start, end))

    pages = {}
    unlabeled_count = 0
    for top_num, start, end in spans:
        segment = text[start:end]
        lines = segment.rstrip("\n").splitlines()
        page_label = None
        last_idx = None

        for idx in range(len(lines) - 1, -1, -1):
            s = lines[idx].strip()
            if not s:
                continue
            m_num = num_re.match(s)
            m_rom = roman_re.match(s) if not m_num else None
            m_alpha = alpha_re.match(s) if not m_num and not m_rom else None

            if m_num:
                page_label = m_num.group(1)
                last_idx = idx
                break
            if m_rom:
                page_label = _normalize_page_label(m_rom.group(1))
                last_idx = idx
                break
            if m_alpha:
                page_label = m_alpha.group(1)
                last_idx = idx
                break
            break

        if not page_label:
            unlabeled_count += 1
            page_label = (
                top_num if top_num is not None else str(10000 * unlabeled_count)
            )
            content = segment.strip()
        else:
            content = "\n".join(lines[:last_idx]).rstrip()
        try:
            key = str(page_to_int(page_label))
        except ValueError:
            print(f"[WARN] Could not parse page label '{page_label}'. Using as-is.")
            key = page_label

        if key in pages:
            pages[key] = pages[key] + "\n\n" + content if content else pages[key]
        else:
            pages[key] = content

    return pages


def pages_by_bottom_number_html(
    raw_html: str, return_html: bool = False
) -> Dict[str, str]:
    """Parse HTML filing into pages based on <hr> tags and page numbers."""
    # Extract only the main document (exclude exhibits which have their own page numbers)
    main_doc = extract_main_document(raw_html)
    html = main_doc.replace("\r\n", "\n").replace("\r", "\n")

    # Strategy 1: Try <hr> tags
    # Simplified to avoid swallowing content between page-break markers and HRs
    hr_re = re.compile(r"<hr\b[^>]*>", re.IGNORECASE)
    parts = hr_re.split(html)

    # Strategy 2: Try <div style="page-break-before
    if len(parts) <= 1:
        page_div_re = re.compile(r"<div[^>]*page-break-before[^>]*>", re.IGNORECASE)
        parts = page_div_re.split(html)

    # Strategy 3: Try <!-- Field: Page --> markers with Value field
    # This is for filings like IntelliCheck that embed page numbers in markers
    if len(parts) <= 1:
        # Look for Field: Page markers with Value field
        page_markers = list(
            re.finditer(r"<!-- Field: Page[^>]*Value:\s*(\d+)[^>]* -->", html)
        )
        if page_markers:
            # Split by Field: Page markers and associate with page numbers
            split_parts = []
            last_pos = 0
            for match in page_markers:
                if match.start() > last_pos:  # Only add if there's content
                    split_parts.append(
                        (html[last_pos : match.start()], int(match.group(1)))
                    )
                last_pos = match.end()
            # Add remaining content
            if last_pos < len(html):
                split_parts.append((html[last_pos:], None))

            split_parts = [
                (seg, page_num) for seg, page_num in split_parts if seg.strip()
            ]

            if len(split_parts) > 1:  # Only use this if we got meaningful splits
                pages = {}
                for segment_html, page_num in split_parts:
                    text = _fast_html_to_text(segment_html)

                    if return_html:
                        content = segment_html
                    else:
                        content = text.strip()

                    if page_num is not None:
                        key = str(page_num)
                    else:
                        key = str(max([int(k) for k in pages.keys()], default=0) + 1)

                    if key in pages:
                        pages[key] = (
                            pages[key] + "\n\n" + content if content else pages[key]
                        )
                    else:
                        pages[key] = content

                return pages

    # Strategy 4: Try simple Field: Page split
    if len(parts) <= 1:
        page_marker_re = re.compile(r"<!-- Field: Page[^>]* -->", re.IGNORECASE)
        parts = page_marker_re.split(html)

    # Fallback: if STILL no page breaks found, treat entire document as one page
    if len(parts) <= 1:
        parts = [html]

    num_re = re.compile(r"^\s*-?(\d{1,4})-?\s*$")
    roman_re = re.compile(r"^\s*-?\(?([IVXLCDMivxlcdm]{1,10})\)?-?\s*$")
    pageword_re = re.compile(r"^\s*(?:page\s+)?-?(\d{1,4})-?\s*$", re.IGNORECASE)
    special_re = re.compile(r"^\s*([A-Z]-\d+)\s*$")

    pages = {}
    for i, segment_html in enumerate(parts, start=1):
        text = _fast_html_to_text(segment_html)
        lines = text.splitlines()

        page_label = None
        last_idx = None

        # Look for page numbers in last 20 lines instead of 16
        for idx in range(len(lines) - 1, max(-1, len(lines) - 20), -1):
            s = lines[idx].strip()
            if not s:
                continue
            m = (
                num_re.match(s)
                or pageword_re.match(s)
                or roman_re.match(s)
                or special_re.match(s)
            )
            if m:
                page_label = _normalize_page_label(m.group(1))
                last_idx = idx
                break
            if len(s) > 100:
                break

        if return_html:
            content = segment_html
        else:
            content = (
                "\n".join(lines[:last_idx]).rstrip() if page_label else text.strip()
            )

        if page_label:
            try:
                key = str(page_to_int(page_label))
            except ValueError:
                # Fallback: use original label as-is
                key = str(page_label)
        else:
            key = str(i)

        if key in pages:
            pages[key] = pages[key] + "\n\n" + content if content else pages[key]
        else:
            pages[key] = content

    return pages


def create_pages_dict(
    raw_content: str, file_extension: str, return_html: bool = False
) -> Dict[str, str]:
    """
    Create a dictionary mapping page numbers to content.

    Args:
        raw_content: The raw filing content
        file_extension: '.txt' or '.htm' or '.html'
        return_html (bool): If True and file is .htm, instructs helper
                            to return HTML pages instead of text.

    Returns:
        Dictionary mapping page numbers (as strings) to page content

    Raises:
        ValueError: If file extension is not supported
    """
    ext = file_extension.lower().lstrip(".")
    if ext == "txt":
        return pages_by_bottom_number(raw_content)
    elif ext in ("htm", "html"):
        return pages_by_bottom_number_html(raw_content, return_html=return_html)
    else:
        raise ValueError(f"Unsupported filing type: {file_extension}")


def _find_section_start_index(
    page_content: str,
    section_name: str,
    return_html: bool = False,
    verbose: bool = False,
) -> int:
    """
    Find the starting index of a section in page content.
    Includes multiple fallback strategies.

    Args:
        page_content: The page text/html to search
        section_name: The section name to find
        return_html: Whether searching in HTML
        verbose: Print debug info

    Returns:
        Starting index or 0 as fallback
    """
    if return_html:
        # For HTML, find all matches and pick the best one
        # Prefer headers with proper formatting (centered, bold, with spacing)

        # Strategy 1: Look for centered bold headers (most reliable)
        # Pattern: <...center...><b>SECTION NAME</b>...
        section_pattern = _build_loose_section_pattern(section_name, allow_tags=True)
        centered_pattern = (
            r'<p[^>]*align="CENTER"[^>]*>.*?<b[^>]*>\s*' + section_pattern + r"\s*</b>"
        )
        match = re.search(centered_pattern, page_content, re.IGNORECASE | re.DOTALL)
        if match:
            if verbose:
                print(
                    f"[CENTERED HEADER] Found '{section_name}' as centered bold header"
                )
            return match.start()

        # Strategy 2: Look for bold headers at the beginning of sections (with proper spacing)
        # Prefer headers that have significant content after them, not just a reference
        bold_pattern = r"<b[^>]*>\s*" + section_pattern + r"\s*</b>"
        bold_matches = list(re.finditer(bold_pattern, page_content, re.IGNORECASE))
        if bold_matches:
            # Prefer bold headers that are on their own (not in the middle of text)
            # and have substantial content after them
            for match in bold_matches:
                before = page_content[max(0, match.start() - 200) : match.start()]
                after = page_content[
                    match.end() : min(len(page_content), match.end() + 400)
                ]

                # Extract text content to check for preamble indicators
                after_text = re.sub("<[^>]+>", " ", after)  # Remove HTML tags
                after_text = re.sub(r"&\w+;", " ", after_text)  # Remove HTML entities
                after_words = after_text.split()

                # Check if surrounded by paragraph tags or newlines (real header)
                # vs in middle of flowing text (reference)
                has_proper_format = (
                    "</p>" in before or "<p" in before or "\n" in before
                ) and (
                    "</b>" in after
                    and ("<p" in after or "\n" in after or "</p>" in after)
                )

                # Check that it's not just a reference (minimal preamble indicators after it)
                preamble_words = [
                    "see",
                    "beginning",
                    "page",
                    "incorporated",
                    "reference",
                ]
                has_preamble = (
                    sum(1 for w in after_words[:30] if w.lower() in preamble_words) >= 2
                )

                if has_proper_format and not has_preamble:
                    if verbose:
                        print(
                            f"[BOLD HEADER] Found '{section_name}' as bold header with proper formatting"
                        )
                    return match.start()

            # Fallback to first bold match if no properly formatted one found
            if verbose:
                print(f"[BOLD MATCH] Found '{section_name}' in bold")
            return bold_matches[0].start()

        # Strategy 3: Standard text match
        pattern = re.compile(section_pattern, re.IGNORECASE)
        matches = list(pattern.finditer(page_content))

        if matches:
            # Find the match that looks most like a header (earliest, with good spacing around it)
            best_match = matches[0]
            for match in matches:
                before = page_content[max(0, match.start() - 50) : match.start()]
                # Prefer matches preceded by newlines or closing tags (not in middle of sentence)
                if before.rstrip().endswith(("\n", ">", "<!-- ")):
                    best_match = match
                    break

            if verbose:
                print(f"[TEXT MATCH] Found '{section_name}'")
            return best_match.start()

        return 0

    # For non-HTML text content
    lines = page_content.split("\n")

    # Strategy 1: Look for a header-like line (normalized, not a TOC entry)
    header_idx = _find_header_line_index(lines, section_name)
    if header_idx is not None:
        if verbose:
            print(f"[STANDALONE HEADER] Found '{section_name}' on own line")
        return sum(len(l) + 1 for l in lines[:header_idx])

    # Strategy 2: Look for patterns that indicate a header
    header_patterns = [
        r"<b[^>]*>[\s\n]*" + re.escape(section_name) + r"[\s\n]*</b>",
        r">[\s\n]*" + re.escape(section_name) + r"[\s\n]*<",
        r"^\s*[\*\-=]+\s*" + re.escape(section_name) + r"\s*[\*\-=]+",
    ]

    for pattern_str in header_patterns:
        match = re.search(pattern_str, page_content, re.IGNORECASE | re.MULTILINE)
        if match:
            if verbose:
                print(f"[HEADER PATTERN] Found section header")
            return match.start()

    # Strategy 3: Standard word sequence match (but prefer matches with good context)
    pattern = _build_loose_section_pattern(section_name)

    # Find all matches and pick the best one (earliest one that's a real header)
    matches = list(re.finditer(pattern, page_content, re.IGNORECASE))
    if matches:
        # Prefer matches where the section name is preceded by minimal content
        best_match = None
        for match in matches:
            # Check if this match looks like a header (preceded by newline or tag)
            before = page_content[max(0, match.start() - 50) : match.start()]
            line_start = page_content.rfind("\n", 0, match.start()) + 1
            line_end = page_content.find("\n", match.start())
            if line_end == -1:
                line_end = len(page_content)
            line = page_content[line_start:line_end]
            if _looks_like_toc_entry_line(line):
                continue
            
            # Check if this is a cross-reference pattern
            after_match = page_content[match.end():match.end() + 50].strip()
            after_upper = after_match.upper()
            
            # Pattern 1: Dash followed by detail (e.g., "Risk Factors - Non-Registration...")
            if after_match and after_match[0] in '-–—':
                after_dash = after_match[1:].strip()
                if len(after_dash) > 3:
                    continue
            
            # Pattern 2: Page reference (e.g., '"RISK FACTORS" BEGINNING AT PAGE 10')
            page_ref_patterns = [
                'BEGINNING AT PAGE',
                'BEGINNING ON PAGE',
                'AT PAGE',
                'ON PAGE',
                '(PAGE',
                '" (PAGE',
                'SEE PAGE',
            ]
            if any(pat in after_upper for pat in page_ref_patterns):
                continue
            
            # Pattern 3: Preceded by "SEE" (e.g., 'SEE "RISK FACTORS"')
            before_text = before.strip().upper()
            if before_text.endswith('SEE') or before_text.endswith('SEE "'):
                continue
            
            if before.endswith("\n") or before.endswith(">") or before.count("\n") >= 1:
                best_match = match
                break

        if best_match is None and matches:
            # Fallback: use first match that isn't a cross-reference
            for match in matches:
                after_match = page_content[match.end():match.end() + 50].strip()
                after_upper = after_match.upper()
                
                # Skip cross-references
                if after_match and after_match[0] in '-–—':
                    after_dash = after_match[1:].strip()
                    if len(after_dash) > 3:
                        continue
                
                page_ref_patterns = [
                    'BEGINNING AT PAGE', 'BEGINNING ON PAGE', 'AT PAGE',
                    'ON PAGE', '(PAGE', '" (PAGE', 'SEE PAGE',
                ]
                if any(pat in after_upper for pat in page_ref_patterns):
                    continue
                    
                best_match = match
                break
        
        if best_match is None and matches:
            best_match = matches[0]  # Last resort fallback
            
        if best_match:
            if verbose:
                print(f"[WORD MATCH] Found '{section_name}'")
            return best_match.start()

    # Strategy 4: Fuzzy match with key words
    key_words = section_name.split()
    best_idx = -1
    best_score = 0

    for i, line in enumerate(lines):
        line_upper = line.upper()
        score = sum(1 for kw in key_words if kw.upper() in line_upper)
        if (
            score > best_score and score >= len(key_words) * 0.6
        ):  # At least 60% of keywords
            best_score = score
            best_idx = i

    if best_idx >= 0:
        if verbose:
            print(f"[FUZZY START] Found section at line {best_idx}")
        return sum(len(line) + 1 for line in lines[:best_idx])

    if verbose:
        print(f"[FALLBACK] Using position 0 for section start")
    return 0


def _find_matching_toc_key(toc: Dict[str, str], section_name: str) -> Optional[str]:
    keys = list(toc.keys())
    normalized_section = _normalize_section_key(section_name)
    if not normalized_section:
        return None

    for k in keys:
        if _normalize_section_key(k) == normalized_section:
            return k

    if HAS_FUZZY:
        matched_section = _fuzzy_match_section(section_name, keys, threshold=75)
        if matched_section:
            return matched_section

    for k in keys:
        normalized_key = _normalize_section_key(k)
        if normalized_section in normalized_key or normalized_key in normalized_section:
            return k

    if normalized_section == "riskfactors":
        for k in keys:
            normalized_key = _normalize_section_key(k)
            if "risk" in normalized_key:
                return k

    return None


def _get_section_header_candidates(toc: Dict[str, str], section_name: str) -> List[str]:
    candidates = [section_name]
    matched_key = _find_matching_toc_key(toc, section_name)
    if matched_key and matched_key not in candidates:
        candidates.append(matched_key)

    normalized_section = _normalize_section_key(section_name)
    if normalized_section == "riskfactors":
        risk_aliases = [
            "Risk Factors",
            "Risks",
            "The Risks You Face",
            "Risks Related to Our Business",
            "Risks Related to Our Company",
            "Risks Associated With Our Company",
            "Risks Associated With This Offering",
            "Risk Factors Relating to Our Company",
            "Risk Factors Relating to Our Business",
            "Risk Factors Relating to Our Common Stock",
            "Risk Factors Relating to this Offering",
            "Summary Information and Risk Factors",
            "Summary and Risk Factors",
        ]
        for alias in risk_aliases:
            if alias not in candidates:
                candidates.append(alias)
        for key in toc.keys():
            normalized_key = _normalize_section_key(key)
            if "risk" in normalized_key and key not in candidates:
                candidates.append(key)

    return candidates


def _find_best_start_index(
    page_content: str,
    candidates: List[str],
    return_html: bool = False,
) -> int:
    best_idx = 0
    best_score = -1
    for candidate in candidates:
        pattern = _build_loose_section_pattern(candidate, allow_tags=return_html)
        if not pattern:
            continue
        if not re.search(pattern, page_content, re.IGNORECASE):
            continue
        idx = _find_section_start_index(
            page_content, candidate, return_html, verbose=False
        )
        score = _score_section_match(page_content[idx:], candidate)
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx


def _page_contains_header(
    page_content: str, candidates: List[str], return_html: bool = False
) -> bool:
    for candidate in candidates:
        pattern = _build_loose_section_pattern(candidate, allow_tags=return_html)
        if pattern and re.search(pattern, page_content, re.IGNORECASE):
            return True
    return False


def _find_section_end_index(
    page_content: str,
    next_section_name: str,
    return_html: bool = False,
    verbose: bool = False,
    min_words: int = 0,
) -> int:
    """
    Find the ending index of a section in page content (where next section starts).
    Includes multiple fallback strategies.

    Args:
        page_content: The page text/html to search
        next_section_name: The next section name to find
        return_html: Whether searching in HTML
        verbose: Print debug info

    Returns:
        Ending index or -1 (use full page) as fallback
    """
    # Strategy 1: Exact match with header patterns
    if not return_html:
        lines = page_content.split("\n")
        target = _normalize_section_key(next_section_name)
        word_counts = []
        total_words = 0
        for line in lines:
            total_words += len(line.split())
            word_counts.append(total_words)

        first_candidate = None
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue
            if _looks_like_toc_entry_line(line_stripped):
                continue
            normalized_line = _normalize_section_key(line_stripped)
            if not normalized_line:
                continue
            if (
                normalized_line == target
                or normalized_line.startswith(target)
                or normalized_line.endswith(target)
            ):
                offset = sum(len(l) + 1 for l in lines[:i])
                words_before = word_counts[i - 1] if i > 0 else 0
                if first_candidate is None:
                    first_candidate = offset
                if min_words <= 0 or words_before >= min_words:
                    if verbose:
                        print(f"[EXACT HEADER END] Found next section header")
                    return offset

        if first_candidate is not None:
            if verbose:
                print(f"[EXACT HEADER END] Found next section header")
            return first_candidate

        patterns = [
            r"(?m)^\s*" + re.escape(next_section_name) + r"\s*$",
            r">[\s\n]*" + re.escape(next_section_name) + r"[\s\n]*<",
            r"<b[^>]*>[\s\n]*" + re.escape(next_section_name) + r"[\s\n]*</b>",
        ]

        for pattern in patterns:
            matches = list(re.finditer(pattern, page_content, re.IGNORECASE))
            for match in matches:
                words_before = len(page_content[: match.start()].split())
                if min_words > 0 and words_before < min_words:
                    continue
                if verbose:
                    print(f"[EXACT HEADER END] Found next section header")
                return match.start()

    # Strategy 1b: Standard word match
    if return_html:
        pattern = re.compile(
            _build_loose_section_pattern(next_section_name, allow_tags=True),
            re.IGNORECASE,
        )
        match = pattern.search(page_content)
        if match:
            return match.start()
    else:
        pattern = _build_loose_section_pattern(next_section_name)
        matches = list(re.finditer(pattern, page_content, re.IGNORECASE))
        for match in matches:
            line_start = page_content.rfind("\n", 0, match.start()) + 1
            line_end = page_content.find("\n", match.start())
            if line_end == -1:
                line_end = len(page_content)
            line = page_content[line_start:line_end]
            if _looks_like_toc_entry_line(line):
                continue
            words_before = len(page_content[: match.start()].split())
            if min_words > 0 and words_before < min_words:
                continue
            return match.start()

    # Strategy 2: Fuzzy match with key words
    if not return_html:
        key_words = next_section_name.split()
        lines = page_content.split("\n")

        for i, line in enumerate(lines):
            line_upper = line.upper()
            score = sum(1 for kw in key_words if kw.upper() in line_upper)
            if score >= len(key_words) * 0.7:  # At least 70% of keywords
                if verbose:
                    print(f"[FUZZY END] Found next section at line {i}")
                return sum(len(l) + 1 for l in lines[:i])

    # Strategy 3: Look for common section header patterns
    if not return_html:
        header_patterns = [
            r"^\s*[\*\-=]+\s*" + re.escape(next_section_name) + r"\s*[\*\-=]+",
            r"^\s*" + re.escape(next_section_name) + r"\s*$",
        ]
        for pattern_str in header_patterns:
            match = re.search(pattern_str, page_content, re.IGNORECASE | re.MULTILINE)
            if match:
                if verbose:
                    print(f"[HEADER PATTERN] Found next section header")
                return match.start()

    # Strategy 4: Common next section keywords fallback
    # If we still haven't found the next section, look for common S-1 sections
    if not return_html:
        common_next_sections = [
            "Prospectus Summary",
            "Risk Factors",
            "Experts",
            "Use of Proceeds",
            "Dividend Policy",
            "Capitalization",
            "Dilution",
            "Selection and Amount of Proceeds",
            "Plans of Distribution",
            "Management's Discussion and Analysis",
            "Business",
            "Description of Securities",
            "Underwriting",
            "Legal Matters",
            "Where You Can Find",
            "Price Range",
            "Market Price",
        ]

        # Try to find any of these common next sections
        for fallback_section in common_next_sections:
            words = list(map(re.escape, fallback_section.upper().split()))
            pattern = r"\s+".join(words)
            match = re.search(pattern, page_content, re.IGNORECASE)
            if match:
                if verbose:
                    print(
                        f"[COMMON SECTION FALLBACK] Found '{fallback_section}' instead"
                    )
                return match.start()

    if verbose:
        print(f"[FALLBACK] Using full page for section end")
    return -1


def _score_section_match(content: str, section_name: str) -> int:
    """Score how likely this is a real section vs. a preamble reference.

    Returns a score where higher scores indicate real sections.
    """
    score = 0
    word_count = len(content.split())

    # Base score from word count (real sections are longer)
    score += min(word_count // 50, 200)  # Max 200 points from word count

    # Penalty for preamble indicators
    first_200 = content[:200].lower()
    preamble_indicators = [
        '". ',
        'see "',
        "beginning on page",
        "agent warrants",
        "underwriter",
    ]

    for indicator in preamble_indicators:
        if indicator in first_200:
            score -= 100

    # Bonus for real section indicators (depends on section name)
    section_lower = section_name.lower()

    if "risk" in section_lower:
        # Risk Factors sections often start with "risks", "risk", "business faces"
        if any(
            word in content[:200].lower()
            for word in [
                "our business faces",
                "business faces",
                "risks",
                "risk",
                "factors",
            ]
        ):
            score += 150

    # Bonus if section name appears on its own line (header format)
    if content.startswith(section_name.upper()) or content.startswith(section_name):
        score += 100

    return max(0, score)


def extract_section_text_html(
    raw_content: str,
    toc: Dict[str, str],
    section_name: str,
    file_extension: str,
    return_html: bool = False,
    verbose: bool = False,
) -> str:
    """
    Extract the text (or HTML) for a specific section.

    Args:
        raw_content: The raw filing content
        toc: Table of contents dictionary {section_name: page_number_as_string}
        section_name: Name of the section to extract
        file_extension: '.txt', '.htm', or '.html'
        return_html (bool): If True, attempt to return raw HTML
                            instead of cleaned plaintext.
        verbose (bool): Print debug messages

    Returns:
        The extracted section text (or HTML)
    """
    try:
        pages_dict = create_pages_dict(
            raw_content, file_extension, return_html=return_html
        )
        header_candidates = _get_section_header_candidates(toc, section_name)
        start, finish = get_section_page_range(toc, section_name, verbose=verbose)

        # STRATEGY: If TOC page doesn't exist and we need to search for it,
        # first try extracting directly from raw_content using pattern matching
        if start not in [int(k) for k in pages_dict.keys() if k.isdigit()]:
            # Try to find the section directly in raw_content
            if verbose:
                print(
                    f"[RAW EXTRACTION] TOC page {start} not in pages_dict, trying raw HTML extraction"
                )

            # Find section header in raw HTML
            matches = []
            for candidate in header_candidates:
                pattern = _build_loose_section_pattern(candidate, allow_tags=True)
                if not pattern:
                    continue
                matches.extend(list(re.finditer(pattern, raw_content, re.IGNORECASE)))

            if matches:
                # Try each match, looking for one that is a proper header (not just a reference)
                candidates = []

                for match in matches:
                    section_start = match.start()

                    # Check if this looks like a header (proper formatting: centered, bold, or on own line)
                    before = raw_content[max(0, section_start - 200) : section_start]
                    after = raw_content[
                        match.end() : min(len(raw_content), match.end() + 200)
                    ]

                    # Heuristics for real header vs reference:
                    # 1. Preceded by <p align="CENTER" or similar
                    # 2. Followed by newline or closing tag
                    # 3. In <b> tag
                    # 4. Not in middle of a sentence

                    is_centered = "<p" in before and "CENTER" in before.upper()
                    is_bold = "<b" in before and "</b>" in after
                    is_new_paragraph = before.rstrip().endswith(
                        "</p>"
                    ) or before.rstrip().endswith(">")
                    is_continuation = ' and "' in before[-50:] or ", " in before[-50:]

                    # Count words after this match
                    section_end = len(raw_content)
                    other_sections = [
                        s for s in toc.keys() if s.lower() != section_name.lower()
                    ]
                    try:
                        for other_sec in other_sections:
                            pattern = _build_loose_section_pattern(
                                other_sec, allow_tags=True
                            )
                            if not pattern:
                                continue
                            next_match = re.search(
                                pattern,
                                raw_content[section_start + 50 :],
                                re.IGNORECASE,
                            )
                            if next_match:
                                potential_end = section_start + 50 + next_match.start()
                                if potential_end > section_start + 500:
                                    section_end = potential_end
                                    break
                    except TypeError:
                        pass  # Ignore matches that cause errors

                    test_content = raw_content[section_start:section_end]
                    test_clean = re.sub("<[^>]+>", " ", test_content)
                    test_clean = re.sub(r"&\w+;", " ", test_clean)
                    test_words = len(test_clean.split())

                    # Score this candidate
                    score = 0
                    if is_centered:
                        score += 30  # Strongly prefer centered
                    if is_bold:
                        score += 20  # Strongly prefer bold
                    if is_new_paragraph:
                        score += 5
                    if not is_continuation:
                        score += 3
                    if test_words > 1000:
                        score += 10
                    if test_words > 5000:
                        score += 15

                    if test_words > 100:  # Only consider if substantial content
                        candidates.append(
                            (score, section_start, section_end, test_words)
                        )

                if candidates:
                    # Sort by score (descending) and pick the best one
                    candidates.sort(key=lambda x: -x[0])
                    best_score, best_start_pos, best_end_pos, best_word_count = (
                        candidates[0]
                    )

                    if verbose:
                        print(
                            f"[RAW SUCCESS] Found '{section_name}' in raw HTML with {best_word_count} words (score={best_score})"
                        )

                    # Extract from raw HTML
                    section_html = raw_content[best_start_pos:best_end_pos]

                    # Clean and return
                    if not return_html:
                        return _fast_html_to_text(section_html)
                    else:
                        return section_html

        collected_pages = []
        first_page = pages_dict.get(str(start))
        found_via_toc = True

        # Fallback: try nearby pages if exact page not found
        if not first_page:
            found_via_toc = False
            # First try +/- 1, 2, 3 pages
            for offset in [1, -1, 2, -2, 3, -3]:
                fallback_page = pages_dict.get(str(start + offset))
                if fallback_page:
                    if verbose:
                        print(
                            f"[FALLBACK] Using page {start + offset} instead of {start}"
                        )
                    first_page = fallback_page
                    break

        # Always do a comprehensive search across all pages when TOC page doesn't exist
        # to find the real section (not just preamble references)
        if not found_via_toc:
            if verbose:
                print(
                    f"[SEARCH ALL] TOC page {start} not found, searching all pages for '{section_name}' header"
                )
            pages_to_search = sorted([int(k) for k in pages_dict.keys() if k.isdigit()])
            best_match = first_page
            best_match_score = 0

            if first_page:
                start_idx = _find_best_start_index(
                    first_page, header_candidates, return_html
                )
                test_remaining = first_page[start_idx:]
                best_match_score = _score_section_match(test_remaining, section_name)

            for search_page in pages_to_search:
                candidate = pages_dict.get(str(search_page))
                if candidate:
                    test_idx = _find_best_start_index(
                        candidate, header_candidates, return_html
                    )
                    test_remaining = candidate[test_idx:]
                    test_score = _score_section_match(test_remaining, section_name)

                    # Prefer matches with higher scores (more likely real section, not preamble reference)
                    if test_score > best_match_score:
                        best_match = candidate
                        best_match_score = test_score
                        if verbose and test_score > 300:
                            print(
                                f"[FOUND IN SEARCH] '{section_name}' on page {search_page} with score {test_score}"
                            )

            if best_match and best_match_score > 100:
                first_page = best_match

        # Check if the first_page actually contains a real section header (not just a reference)
        # If it only has a preamble reference, search other pages
        if first_page:
            start_idx = _find_best_start_index(
                first_page, header_candidates, return_html
            )
            remaining_content = first_page[start_idx:]
            remaining_words = len(remaining_content.split())

            # Detect if this is a preamble reference (common patterns):
            # 1. Very few words after the match (< 100 words is suspicious)
            # 2. Content immediately after section name is a quote or reference pattern
            # 3. For Risk Factors specifically, real section usually starts with "business faces" or "business" description
            is_preamble_reference = False
            preamble_reason = ""

            # More aggressive detection: if remaining content is too short, it's likely a reference
            if remaining_words < 100:
                is_preamble_reference = True
                preamble_reason = f"too short ({remaining_words} words)"
                if verbose:
                    print(
                        f"[PREAMBLE DETECTION] Very short content ({remaining_words} words) suggests preamble reference"
                    )
            elif (
                remaining_words < 2000
            ):  # Check for preamble patterns with higher threshold
                remaining_upper = remaining_content.upper()
                remaining_first_600 = remaining_content[:600]  # Get more context

                # Common preamble patterns - be very specific
                preamble_patterns = [
                    (
                        r'"\s*\.\s+(?:Investing|You|We)',
                        "Quoted section ending followed by new sentence",
                    ),
                    (
                        r'(?i)See\s+"[^"]*"\s+beginning on page',
                        "See X beginning on page pattern",
                    ),
                    (
                        r"(?i)See\s+Risk\s+Factors\s+beginning on page",
                        "See Risk Factors beginning on page pattern",
                    ),
                    (
                        r"(?i)incorporated\s+by\s+reference",
                        "Incorporated by reference reference",
                    ),
                    (
                        r"(?i)the risks described in",
                        "Reference to risks described elsewhere",
                    ),
                ]

                for pattern, reason in preamble_patterns:
                    if re.search(pattern, remaining_first_600):
                        is_preamble_reference = True
                        preamble_reason = reason
                        if verbose:
                            print(
                                f"[PREAMBLE DETECTION] Found preamble indicator: {reason}"
                            )
                        break

            if is_preamble_reference and len(pages_dict) > 1:
                if verbose:
                    print(
                        f"[CHECK] Page {start} appears to have preamble reference ({remaining_words} words, reason: {preamble_reason}), checking if there's substantial content on this or following pages"
                    )

                # Many filings have: Header + preamble intro + real content all on same page
                # So we check: does this page have enough content overall? If yes, KEEP IT and collect subsequent pages
                # Only search for alternative pages if this page is VERY short (< 150 words total)

                if remaining_words >= 150:
                    # This page has substantial content - keep it and collect subsequent pages
                    if verbose:
                        print(
                            f"[DECISION] Page {start} has {remaining_words} words after header - KEEPING this page and collecting all subsequent pages"
                        )
                    # Continue with normal collection logic (don't replace first_page)
                    found_real_header = True
                else:
                    # Page is too short - search for alternative pages with more content
                    if verbose:
                        print(
                            f"[DECISION] Page {start} only has {remaining_words} words - searching for pages with more substantial content"
                        )

                    # Search through all pages for the real section
                    # Prioritize pages that actually contain the section header text
                    pages_to_search = sorted(
                        [int(k) for k in pages_dict.keys() if k.isdigit()]
                    )
                    found_real_header = False
                    candidates = []

                    for search_page in pages_to_search:
                        # Skip the original "first_page" number since we already know it's just a preamble
                        if search_page == start:
                            continue

                        candidate_page = pages_dict.get(str(search_page))
                        if candidate_page:
                            # First check if this page actually contains the section header
                            if not _page_contains_header(
                                candidate_page, header_candidates, return_html
                            ):
                                continue  # Skip pages that don't have the section name at all

                            # Check if this page is just a TOC/cover page (contains multiple section headers from toc)
                            # TOC pages typically mention many sections
                            toc_section_count = 0
                            for toc_section_name in toc.keys():
                                if (
                                    toc_section_name.lower() != section_name.lower()
                                ):  # Don't count current section
                                    toc_pattern = _build_loose_section_pattern(
                                        toc_section_name, allow_tags=return_html
                                    )
                                    if toc_pattern and re.search(
                                        toc_pattern, candidate_page, re.IGNORECASE
                                    ):
                                        toc_section_count += 1

                            # If page contains 2+ other section names, it's likely a TOC page - skip it
                            if toc_section_count >= 2:
                                if verbose:
                                    print(
                                        f"[SKIP] Page {search_page} appears to be TOC page (contains {toc_section_count} other section headers)"
                                    )
                                continue

                            test_idx = _find_section_start_index(
                                candidate_page, section_name, return_html, verbose=False
                            )
                            test_remaining = candidate_page[test_idx:]
                            test_words = len(test_remaining.split())

                            # Skip pages with too little content, but be lenient for preamble references
                            # which may legitimately be just a few sentences in S-1 filings
                            min_words_threshold = 100 if is_preamble_reference else 300
                            if test_words < min_words_threshold:
                                continue

                            # Check if this looks like a preamble reference
                            test_first_600 = test_remaining[:600]
                            test_preamble_patterns = [
                                (
                                    r'"\s*\.\s+(?:Investing|You|We)',
                                    "Quoted section ending",
                                ),
                                (
                                    r'(?i)See\s+"[^"]*"\s+beginning on page',
                                    "See X beginning on page",
                                ),
                                (
                                    r"(?i)See\s+Risk\s+Factors\s+beginning on page",
                                    "See Risk Factors beginning on page",
                                ),
                                (
                                    r"(?i)incorporated\s+by\s+reference",
                                    "Incorporated by reference",
                                ),
                                (
                                    r"(?i)the risks described in",
                                    "Reference to risks described elsewhere",
                                ),
                            ]

                            is_test_preamble = any(
                                re.search(pattern, test_first_600)
                                for pattern, _ in test_preamble_patterns
                            )

                            # Score this candidate
                            score = test_words  # Longer content is better

                            # If it's a preamble reference, heavily penalize it
                            if is_test_preamble:
                                score = score // 10

                            # If content starts with actual section text (good indicators), boost score
                            good_starts = [
                                "our business faces many",
                                "our business",
                                "we face",
                                "the company faces",
                                "risks and uncertainties",
                                "you should carefully consider",
                            ]

                            test_first_100 = test_remaining[:100].lower()
                            if any(start in test_first_100 for start in good_starts):
                                score = score * 2

                            candidates.append(
                                (score, search_page, candidate_page, test_words)
                            )

                    # Use the highest-scoring candidate
                    if candidates:
                        candidates.sort(key=lambda x: -x[0])
                        _, best_page, best_candidate, best_words = candidates[0]
                        first_page = best_candidate
                        if verbose:
                            print(
                                f"[FOUND] Selected page {best_page} with {best_words} words (score={candidates[0][0]})"
                            )
                        found_real_header = True
                    elif len(candidates) == 0:
                        # No candidates found - likely due to poor text extraction (e.g., old HTML formats)
                        # Try raw HTML extraction as fallback
                        if verbose:
                            print(
                                f"[RAW EXTRACTION FALLBACK] No good candidates found in text extraction, trying raw HTML search..."
                            )

                        # Find section directly in raw_content using HTML pattern matching
                        matches = []
                        for candidate in header_candidates:
                            pattern = _build_loose_section_pattern(
                                candidate, allow_tags=True
                            )
                            if not pattern:
                                continue
                            matches.extend(
                                list(re.finditer(pattern, raw_content, re.IGNORECASE))
                            )

                        if matches:
                            # Try each match, looking for one that is a proper header (not just a reference)
                            best_raw_start = None
                            best_raw_words = 0

                            for match in matches:
                                section_start = match.start()
                                before = raw_content[
                                    max(0, section_start - 200) : section_start
                                ]
                                after = raw_content[
                                    match.end() : min(
                                        len(raw_content), match.end() + 200
                                    )
                                ]

                                # Heuristics for real header vs reference:
                                # Check for bold tags around the matched section name
                                is_bold = "<b" in before and "</b>" in after
                                # Check if it's in a centered or aligned paragraph
                                is_centered = (
                                    "<p" in before and "CENTER" in before.upper()
                                )
                                # Check if it's at the start of a new paragraph
                                is_new_paragraph = before.rstrip().endswith("</p>")
                                # Check for <FONT> tag wrapper (old MS Word HTML format)
                                is_font_bold = (
                                    "<font" in before.lower()
                                    and "<b>" in before.lower()
                                    and "</b>" in after.lower()
                                )

                                if (
                                    is_centered
                                    or is_bold
                                    or is_new_paragraph
                                    or is_font_bold
                                ):
                                    # This looks like a header, check for content after it
                                    next_section_pos = len(raw_content)
                                    for other_sec in toc.keys():
                                        if other_sec.lower() != section_name.lower():
                                            pattern = _build_loose_section_pattern(
                                                other_sec, allow_tags=True
                                            )
                                            if not pattern:
                                                continue
                                            next_match = re.search(
                                                pattern,
                                                raw_content[section_start + 50 :],
                                                re.IGNORECASE,
                                            )
                                            if next_match:
                                                potential_pos = (
                                                    section_start
                                                    + 50
                                                    + next_match.start()
                                                )
                                                if potential_pos > section_start + 500:
                                                    next_section_pos = potential_pos
                                                    break

                                    test_html = raw_content[
                                        section_start:next_section_pos
                                    ]
                                    test_text = _fast_html_to_text(test_html)
                                    test_words = len(test_text.split())

                                    if test_words > best_raw_words:
                                        best_raw_start = section_start
                                        best_raw_words = test_words

                            if best_raw_start is not None:
                                if verbose:
                                    print(
                                        f"[RAW EXTRACTION SUCCESS] Found '{section_name}' in raw HTML with {best_raw_words} words"
                                    )

                                # Use raw HTML for extraction
                                next_section_pos = len(raw_content)
                                for other_sec in toc.keys():
                                    if other_sec.lower() != section_name.lower():
                                        pattern = _build_loose_section_pattern(
                                            other_sec, allow_tags=True
                                        )
                                        if not pattern:
                                            continue
                                        next_match = re.search(
                                            pattern,
                                            raw_content[best_raw_start + 50 :],
                                            re.IGNORECASE,
                                        )
                                        if next_match:
                                            potential_pos = (
                                                best_raw_start + 50 + next_match.start()
                                            )
                                            if potential_pos > best_raw_start + 500:
                                                next_section_pos = potential_pos
                                                break

                                section_html = raw_content[
                                    best_raw_start:next_section_pos
                                ]
                                section_text = _fast_html_to_text(section_html)

                                if verbose:
                                    print(
                                        f"[RAW EXTRACTION] Returning {len(section_text.split())} words from raw HTML"
                                    )
                                return section_text

        if not first_page:
            # Ultimate fallback: if we have no page breaks in document, use all available content
            if len(pages_dict) == 1:
                first_page = pages_dict[list(pages_dict.keys())[0]]
                if verbose:
                    print(
                        f"[FALLBACK] Using only available page (no page breaks detected)"
                    )

        if not first_page:
            if verbose:
                print(
                    f"[ERROR] Could not find start page {start} or nearby pages for section '{section_name}'."
                )
            return ""

        start_index = _find_section_start_index(
            first_page, section_name, return_html, verbose
        )

        first_page_content = first_page[start_index:]
        collected_pages.append(first_page_content)

        # Get list of all available pages
        available_pages = sorted([int(k) for k in pages_dict.keys() if k.isdigit()])
        max_page = max(available_pages) if available_pages else finish

        # Collect pages aggressively: collect all pages from start onwards
        # We'll search for the next section and stop when we find it
        # This handles both sequential and sparse page numbering
        for page_num in available_pages:
            if page_num > start:  # Collect all pages after start page
                page_content = pages_dict.get(str(page_num))
                if page_content:
                    collected_pages.append(page_content)

        sections = list(toc.keys())
        next_header_idx = -1

        # Try to find the current section in the TOC keys
        for idx, sec in enumerate(sections):
            if sec.lower().strip() == section_name.lower().strip():
                next_header_idx = idx + 1
                break

        # Fallback: use fuzzy matching if exact match not found
        if next_header_idx == -1 and HAS_FUZZY:
            matched_section = _fuzzy_match_section(section_name, sections, threshold=75)
            if matched_section:
                next_header_idx = sections.index(matched_section) + 1

        if 0 <= next_header_idx < len(sections):
            next_section = sections[next_header_idx]

            # Look for next section boundary in collected pages
            collected_text = "\n\n".join(collected_pages)

            min_words_for_end = 0
            if "risk" in section_name.lower():
                page_span = max(1, min(10, finish - start)) if finish and start else 1
                min_words_for_end = max(400, min(2000, page_span * 200))

            end_index = _find_section_end_index(
                collected_text,
                next_section,
                return_html,
                verbose,
                min_words=min_words_for_end,
            )

            if end_index != -1:
                # Found the next section boundary
                if verbose:
                    print(
                        f"[FOUND BOUNDARY] Next section '{next_section}' found in collected pages"
                    )
                collected_text = collected_text[:end_index]

            collected_pages = [collected_text]
        else:
            # No next section found, use all collected pages
            if verbose:
                print(
                    f"[NO NEXT SECTION] Using all {len(collected_pages)} collected pages"
                )

        joined_content = "\n\n".join(collected_pages)
        if return_html:
            return joined_content.strip()
        else:
            return re.sub(r"\s+", " ", joined_content).strip()

    except (KeyError, ValueError, Exception) as e:
        if verbose:
            print(f"[ERROR] Failed to extract section '{section_name}': {e}")
            import traceback

            traceback.print_exc()
        return ""


def extract_section_text_ascii(
    raw_content: str,
    toc: Dict[str, str],
    section_name: str,
    file_extension: str,
    verbose: bool = False,
) -> str:
    """
    Extract the text (or HTML) for a specific section.

    Args:
        raw_content: The raw filing content
        toc: Table of contents dictionary {section_name: page_number_as_string}
        section_name: Name of the section to extract
        file_extension: '.txt', '.htm', or '.html'
        return_html (bool): If True, attempt to return raw HTML
                            instead of cleaned plaintext.

    Returns:
        The extracted section text (or HTML)
    """
    # Flexible whitespace separator that allows multi-line headers
    # but limits to max 3 newlines to prevent matching across unrelated sections
    sep = r"(?:[ \t]+|(?:\r?\n[ \t]*){1,3})"
    try:
        toc_sections = {section.strip().lower(): section for section in toc.keys()}
        if (
            section_name.lower().strip() == "prospectus summary"
            and "summary" in toc_sections
        ):
            section_name = toc_sections["summary"]
        if (
            section_name.lower().strip() == "summary"
            and "prospectus summary" in toc_sections
        ):
            section_name = toc_sections["prospectus summary"]
        start, finish = get_section_page_range(toc, section_name)
        pages_dict = create_pages_dict(raw_content, file_extension)

        collected_pages = []
        first_page = pages_dict.get(str(start))

        # Sanity check: if start page is invalid (e.g., 30000) or range is inverted,
        # the TOC parsing probably failed. Use fallback.
        if start > finish or start > 1000:  # Page numbers > 1000 are suspicious
            if verbose:
                print(
                    f"[WARN] Invalid page range ({start} to {finish}) for section '{section_name}'. Using fallback..."
                )
            return ascii_extraction_fallback(raw_content, toc, section_name, verbose)

        if not first_page:
            if verbose:
                print(
                    f"[WARN] Could not find start page {start} for section '{section_name}'. Attempting fallback..."
                )
            return ascii_extraction_fallback(raw_content, toc, section_name, verbose)

        # Use _find_section_start_index which has smart detection for cross-references
        # vs real section headers
        start_index = _find_section_start_index(
            first_page, section_name, return_html=False, verbose=verbose
        )

        if start_index == 0:
            # _find_section_start_index returns 0 as fallback, check if it's valid
            # Try the original pattern matching as backup
            words = list(map(re.escape, section_name.split()))
            upper = r"\s+".join(w.upper() for w in words)
            title = r"\s+".join(
                w.capitalize() if i == 0 or w.lower() not in STOPWORDS else w.lower()
                for i, w in enumerate(words)
            )
            upper_parts = upper.split(r"\s+")
            title_parts = title.split(r"\s+")
            upper_pattern = sep.join(upper_parts)
            title_pattern = sep.join(title_parts)
            pattern = rf"(?:{upper_pattern}|{title_pattern})"
            match = re.search(pattern, first_page)
            if match:
                start_index = match.start()

        if start_index == -1 or (start_index == 0 and section_name.lower() not in first_page[:200].lower()):
            if verbose:
                print(
                    f"[WARN] Could not find start of section '{section_name}' in page {start}. Attempting fallback..."
                )
            fallback = ascii_extraction_fallback(raw_content, toc, section_name, verbose)
            if fallback != "":
                return fallback
            print(f"[WARN] Fallback failed, using full starting page")
            start_index = 0

        first_page_content = first_page[start_index:]
        
        # Check if this is a placeholder section (e.g., "RISK FACTORS  See 'Risk Factors' beginning on page 1")
        # These are common in prospectus supplements that reference the main prospectus
        placeholder_pattern = r'^\s*See\s*["\']?' + re.escape(section_name.split()[0])
        # Get text after the section header (skip header itself)
        header_end = re.search(r'\n', first_page_content)
        content_after_header = first_page_content[header_end.end():] if header_end else first_page_content[50:]
        is_placeholder = re.search(placeholder_pattern, content_after_header[:200], re.IGNORECASE)
        
        if is_placeholder:
            if verbose:
                print(f"[DEBUG] Found placeholder section (says 'See {section_name.split()[0]}...'). Using fallback...")
            fallback = ascii_extraction_fallback(raw_content, toc, section_name, verbose)
            if fallback and len(fallback.split()) > len(first_page_content.split()) // 2:
                return fallback
            # If fallback didn't find more content, continue with the placeholder

        collected_pages.append(first_page_content)
        for i in range(start + 1, finish):
            page_content = pages_dict.get(str(i))
            if page_content:
                collected_pages.append(page_content)

        sections = list(toc.keys())
        section_header_idx = -1
        try:
            if section_name in sections:
                section_header_idx = sections.index(section_name)
            else:
                # Try normalized matching
                for idx, sec in enumerate(sections):
                    if _normalize_section_key(sec) == _normalize_section_key(section_name):
                        section_header_idx = idx
                        break
                if section_header_idx == -1:
                    raise ValueError()
        except ValueError:
            if verbose:
                print(
                    f"[WARN] Section '{section_name}' not found in TOC keys. Cannot find end of section."
                )

        if section_header_idx >= 0:
            # Find the next MAJOR section, skipping subsections
            next_section = _find_next_major_section(toc, section_name, section_header_idx)
            
            if not next_section:
                # No next section found - we're at the end, use all remaining pages
                if verbose:
                    print(f"[DEBUG] No next major section found after '{section_name}'")
                joined_content = "\n\n".join(collected_pages)
                return re.sub(r"\s+", " ", joined_content).strip()
            
            if verbose:
                print(f"[DEBUG] Looking for next major section '{next_section}' to determine endpoint")
            
            # Get the final page AND a few extra pages beyond
            # TOC page numbers often don't exactly match where headers appear
            # (e.g., TOC says "Use of Proceeds: 14" but header is actually on page 15)
            extra_pages_to_check = 3  # Check up to 3 pages beyond the TOC-indicated end
            extended_pages = []
            for extra_page_num in range(finish, finish + extra_pages_to_check + 1):
                extra_page = pages_dict.get(str(extra_page_num))
                if extra_page:
                    extended_pages.append(extra_page)
            
            final_page = "\n\n".join(extended_pages) if extended_pages else None
            
            if final_page:
                if verbose:
                    print(f"[DEBUG] Searching for '{next_section}' in combined content (pages {start+1}-{finish+extra_pages_to_check})")
                
                # Use flexible pattern that handles optional words like "the", "of"
                # This catches cases like TOC: "Price Range of the Common Stock" vs header: "PRICE RANGE OF COMMON STOCK"
                base_pattern = _build_flexible_section_pattern(next_section)
                
                # Search in collected pages PLUS the final page to handle cases where
                # the next section appears in the middle of an already-collected page
                joined_with_final = "\n\n".join(collected_pages + [final_page])
                
                # First, find ALL matches of the section name
                all_matches = list(re.finditer(base_pattern, joined_with_final))
                
                # Find a match that appears as a section HEADER (on its own line, possibly indented)
                # A section header typically:
                # 1. Appears after a newline followed only by whitespace (indentation)
                # 2. Is NOT part of a sentence (not preceded by text like "See ")
                # 3. Is in ALL CAPS (most reliable indicator), OR
                # 4. Is standalone on its own line (nothing meaningful after it)
                # 
                # IMPORTANT: has_significant_indent alone is NOT enough because
                # SEC filings often have indented paragraphs where inline text like
                # "The Company currently anticipates..." starts at indent level.
                # We must require ALL CAPS or standalone status.
                
                def is_possessive_or_continuation(text, match_end):
                    """Check if text after match indicates this is NOT a section header.
                    Returns True if it's a possessive ('s) or continuation of a phrase."""
                    after = text[match_end:match_end + 5]
                    # Check for possessive markers: 's, 'S, '
                    if after.startswith("'s") or after.startswith("'S") or after.startswith("'"):
                        return True
                    # Check for comma continuation (e.g., "THE COMPANY, INC.")
                    if after.startswith(",") or after.startswith("."):
                        return True
                    return False
                
                match = None
                # First pass: look for ALL CAPS matches (most reliable)
                for m in all_matches:
                    line_start = joined_with_final.rfind('\n', 0, m.start())
                    if line_start == -1:
                        line_start = 0
                    else:
                        line_start += 1
                    
                    text_before_on_line = joined_with_final[line_start:m.start()]
                    
                    if text_before_on_line.strip() == '':
                        matched_text = joined_with_final[m.start():m.end()]
                        is_all_caps = matched_text.upper() == matched_text
                        
                        # Skip possessives like "THE COMPANY'S MANAGEMENT"
                        if is_possessive_or_continuation(joined_with_final, m.end()):
                            continue
                        
                        # NEW: Skip if followed by more ALL CAPS text on same line
                        # This catches cases like "BUSINESS RISKS" where we match "BUSINESS"
                        # but this is a subsection header, not the "Business" section
                        line_end = joined_with_final.find('\n', m.end())
                        if line_end == -1:
                            line_end = len(joined_with_final)
                        text_after_on_line = joined_with_final[m.end():line_end].strip()
                        
                        if text_after_on_line:
                            # If text after is also ALL CAPS, this is likely a longer subsection header
                            # e.g., "BUSINESS RISKS" where we matched "BUSINESS"
                            first_word_after = text_after_on_line.split()[0] if text_after_on_line.split() else ''
                            if first_word_after and first_word_after.upper() == first_word_after and len(first_word_after) >= 3:
                                # Skip this match - it's part of a longer all-caps header
                                continue
                        
                        if is_all_caps:
                            match = m
                            break
                
                # Second pass: look for standalone headers (nothing after on line)
                if match is None:
                    for m in all_matches:
                        line_start = joined_with_final.rfind('\n', 0, m.start())
                        if line_start == -1:
                            line_start = 0
                        else:
                            line_start += 1
                        
                        text_before_on_line = joined_with_final[line_start:m.start()]
                        
                        line_end = joined_with_final.find('\n', m.end())
                        if line_end == -1:
                            line_end = len(joined_with_final)
                        text_after_on_line = joined_with_final[m.end():line_end]
                        
                        if text_before_on_line.strip() == '':
                            is_standalone = text_after_on_line.strip() == ''
                            
                            if is_standalone:
                                match = m
                                break
                
                if verbose:
                    print(f"[DEBUG] Found {len(all_matches)} total matches for '{next_section}'")
                    if match:
                        print(f"[DEBUG] Using match at position {match.start()} (appears as section header)")
                        context_start = max(0, match.start() - 100)
                        context_end = min(len(joined_with_final), match.start() + 100)
                        print(f"[DEBUG] Context: ...{repr(joined_with_final[context_start:context_end])}...")
                    else:
                        if all_matches:
                            print(f"[DEBUG] None of the matches appear as section headers (all are inline references)")
                            for i, m in enumerate(all_matches[:3]):
                                line_start = joined_with_final.rfind('\n', 0, m.start())
                                if line_start == -1:
                                    line_start = 0
                                else:
                                    line_start += 1
                                text_before = joined_with_final[line_start:m.start()]
                                print(f"[DEBUG]   Match {i+1} at {m.start()}, text before on same line: {repr(text_before[-50:])}")
                        else:
                            print(f"[DEBUG] No matches found at all!")
                
                if match is not None:
                    # Found the boundary - use content up to that point
                    if verbose:
                        print(f"[DEBUG] ✓ Found '{next_section}' at position {match.start()} in combined content")
                    joined_content = joined_with_final[:match.start()]
                    
                    # Check for common "proper ending" patterns that indicate the actual section end
                    # These patterns typically appear right before boilerplate registration content
                    # and signal the true end of a section's substantive content
                    # Note: We look for these followed by whitespace/newlines (not end of string $)
                    proper_ending_patterns = [
                        # "See 'Section Name'" patterns indicating cross-reference at end
                        # Followed by newlines or end of content
                        r'See\s+"[^"]+"\s*\.?(?=\s*\n|\s*$)',
                        r"See\s+'[^']+'\s*\.?(?=\s*\n|\s*$)",
                        # "See 'Section Name' and 'Other Section'" patterns
                        r'See\s+"[^"]+"\s+and\s+"[^"]+"\s*\.?(?=\s*\n|\s*$)',
                        r"See\s+'[^']+'\s+and\s+'[^']+'\s*\.?(?=\s*\n|\s*$)",
                    ]
                    
                    # Look for these patterns in the last portion of the content
                    # Check the last 25% of content (to find ending near the end, not random matches)
                    search_start = int(len(joined_content) * 0.75)
                    search_portion = joined_content[search_start:]
                    
                    best_ending_pos = None
                    best_ending_match = None
                    for pattern in proper_ending_patterns:
                        # Search from the end backwards - we want the LAST proper ending
                        matches_in_tail = list(re.finditer(pattern, search_portion, re.IGNORECASE))
                        if matches_in_tail:
                            # Use the last match (closest to where we'd cut off)
                            last_match = matches_in_tail[-1]
                            ending_pos = search_start + last_match.end()
                            if best_ending_pos is None or ending_pos > best_ending_pos:
                                best_ending_pos = ending_pos
                                best_ending_match = search_portion[last_match.start():last_match.end()]
                    
                    # If we found a proper ending, truncate there to remove boilerplate
                    if best_ending_pos is not None:
                        if verbose:
                            removed_chars = len(joined_content) - best_ending_pos
                            print(f"[DEBUG] Found proper ending pattern: {repr(best_ending_match)}")
                            print(f"[DEBUG] Truncating at proper ending, removing {removed_chars} chars of boilerplate")
                        joined_content = joined_content[:best_ending_pos]
                    
                    if verbose:
                        print(f"[DEBUG] Returning {len(joined_content)} chars (before whitespace normalization)")
                    return re.sub(r"\s+", " ", joined_content).strip()
                else:
                    # Still can't find it - just use pages before the final page
                    # This is safer than including potentially wrong content from final page
                    if verbose:
                        print(f"[WARN] Next section '{next_section}' not found even in combined content. Using pages {start}-{finish-1} only.")
                    joined_content = "\n\n".join(collected_pages)
                    return re.sub(r"\s+", " ", joined_content).strip()
            else:
                if verbose:
                    print(f"[WARN] Could not find final page {finish}")
                # Just use collected pages
                joined_content = "\n\n".join(collected_pages)
                return re.sub(r"\s+", " ", joined_content).strip()

        # Fallback: no section_header_idx found or other edge case
        joined_content = "\n\n".join(collected_pages)
        return re.sub(r"\s+", " ", joined_content).strip()

    except (KeyError, Exception, ValueError) as e:
        if str(e).startswith("Unsupported page label"):
            print(
                "[WARN] Section extraction main method failed. Attempting fallback..."
            )
            return ascii_extraction_fallback(raw_content, toc, section_name, verbose)
        if verbose:
            print(f"[ERROR] Failed to extract section '{section_name}': {e}")
        return ""


def ascii_extraction_fallback(content, toc, section_name, verbose):
    # Flexible whitespace separator that allows multi-line headers
    # but limits to max 3 newlines to prevent matching across unrelated sections
    sep = r"(?:[ \t]+|(?:\r?\n[ \t]*){1,3})"
    
    # Cross-reference patterns to skip
    cross_ref_patterns = [
        'BEGINNING AT PAGE', 'BEGINNING ON PAGE', 'AT PAGE', 'ON PAGE',
        '(PAGE', '" (PAGE', 'SEE PAGE',
    ]
    
    def is_cross_reference_or_toc(match_pos, text, matched_text_len):
        """Check if a match is a cross-reference or TOC entry rather than a real header."""
        # Check text after the match
        after_text = text[match_pos:match_pos + 100].upper()
        
        # Skip if followed by page reference patterns
        for pattern in cross_ref_patterns:
            if pattern in after_text[:50]:
                return True
        
        # Check if preceded by "SEE" or similar cross-reference language
        before_start = max(0, match_pos - 50)
        before_text = text[before_start:match_pos].upper().strip()
        if before_text.endswith('SEE') or before_text.endswith('SEE "'):
            return True
        
        # Check for "the heading" or "under" or "captioned" patterns before quoted section name
        # These indicate cross-references like: under the heading "Risk Factors"
        cross_ref_before_patterns = [
            'THE HEADING "', 'HEADING "', 'UNDER "', 'CAPTIONED "', 
            'TITLED "', 'ENTITLED "', 'UNDER THE "', 'IN THE "',
            'REFER TO "', 'REFERRED TO AS "', 'SET FORTH IN "'
        ]
        for pattern in cross_ref_before_patterns:
            if before_text.endswith(pattern):
                return True
        
        # Check if followed by dash and detail (cross-reference like "Risk Factors - Topic")
        after_stripped = text[match_pos + matched_text_len:match_pos + matched_text_len + 50].strip()
        if after_stripped and after_stripped[0] in '-–—':
            after_dash = after_stripped[1:].strip()
            if len(after_dash) > 3:
                return True
        
        # NEW: Check if this is a TOC entry (surrounded by dots, or preceded by numbers like "3.")
        # Look at text before the match on the same line
        line_start = text.rfind('\n', 0, match_pos)
        if line_start == -1:
            line_start = 0
        else:
            line_start += 1
        text_before_on_line = text[line_start:match_pos]
        
        # Check if preceded by TOC-style numbering (e.g., "3.", "Item 3.", "III.")
        toc_number_pattern = r'(?:^|item\s*)?\d+\.?\s*$|^[ivxIVX]+\.\s*$'
        if re.search(toc_number_pattern, text_before_on_line.strip(), re.IGNORECASE):
            return True
        
        # Check for dots/leaders after the section name (common in TOC)
        after_match = text[match_pos + matched_text_len:match_pos + matched_text_len + 80]
        # TOC entries often have "...." or "......" followed by page numbers or section references
        if re.search(r'^\s*\.{3,}', after_match):
            return True
        
        # Check if this appears to be in a cross-reference table (multiple dots on same line)
        line_end = text.find('\n', match_pos)
        if line_end == -1:
            line_end = len(text)
        full_line = text[line_start:line_end]
        # If the line has repeated dots pattern (like "......." or ". . . . ."), it's a TOC line
        if len(re.findall(r'\.{2,}', full_line)) >= 1 or len(re.findall(r'\. \. \.', full_line)) >= 1:
            return True
        
        # NEW: Check if after text is just whitespace followed by a page number (TOC entry)
        # Example: "RISK FACTORS                                                                 8"
        after_on_line = text[match_pos + matched_text_len:line_end]
        if re.match(r'^\s+\d{1,4}\s*$', after_on_line):
            return True
        
        # NEW: Also check if the NEXT line is just a page number (TOC with page on separate line)
        # Example: "RISK FACTORS\n4\n..."
        # Only check if the current line has nothing after the section name
        if after_on_line.strip() == '':
            next_line_end = text.find('\n', line_end + 1)
            if next_line_end == -1:
                next_line_end = len(text)
            next_line = text[line_end + 1:next_line_end]
            if re.match(r'^\s*\d{1,4}\s*$', next_line.strip()):
                return True
        
        return False
    
    def is_standalone_header(match_pos, text, matched_text_len):
        """Check if a match appears to be a standalone section header (not inline)."""
        # Find line boundaries
        line_start = text.rfind('\n', 0, match_pos)
        if line_start == -1:
            line_start = 0
        else:
            line_start += 1
        line_end = text.find('\n', match_pos + matched_text_len)
        if line_end == -1:
            line_end = len(text)
        
        text_before = text[line_start:match_pos]
        text_after = text[match_pos + matched_text_len:line_end]
        
        # Standalone headers have mostly whitespace before and after
        # Or are preceded by <PAGE> marker
        has_page_marker = '<PAGE>' in text[max(0, match_pos - 100):match_pos]
        is_mostly_whitespace_before = text_before.strip() == '' or len(text_before.strip()) <= 5
        is_mostly_whitespace_after = text_after.strip() == '' or text_after.strip() in ['.', ':', '-']
        
        return (has_page_marker or is_mostly_whitespace_before) and is_mostly_whitespace_after
    
    def is_placeholder_section(match_pos, text, matched_text_len, section_name):
        """Check if this is a placeholder section that just says 'See X' without real content."""
        # Get the text after the header (next 200 chars after whitespace)
        after_header = text[match_pos + matched_text_len:match_pos + matched_text_len + 200]
        # Skip whitespace to get to content
        after_stripped = after_header.strip()[:150]
        
        # Check if content starts with "See X" pattern pointing to the same section
        # E.g., header "RISK FACTORS" followed by 'See "Risk Factors"'
        see_pattern = re.search(r'^See\s*["\']?' + re.escape(section_name.split()[0]), after_stripped, re.IGNORECASE)
        if see_pattern:
            # Check if this is a short placeholder (< 100 chars of real content before next section)
            # Real sections have substantial content
            remaining = after_stripped[see_pattern.end():]
            # If remaining text is mostly just references and short sentences, it's a placeholder
            if len(remaining) < 100 or remaining.count('.') <= 2:
                return True
        
        return False
    
    def find_first_real_header_match(matches, text, section_name=None):
        """Find the first match that is a real section header (not cross-ref or TOC)."""
        for match in matches:
            matched_len = match.end() - match.start()
            if is_cross_reference_or_toc(match.start(), text, matched_len):
                continue
            # Check if standalone header
            if is_standalone_header(match.start(), text, matched_len):
                # Additional check: skip placeholder sections (e.g., "RISK FACTORS  See Risk Factors...")
                if section_name and is_placeholder_section(match.start(), text, matched_len, section_name):
                    continue
                return match.start()
        
        # If no standalone header found, return first non-cross-reference match
        for match in matches:
            matched_len = match.end() - match.start()
            if not is_cross_reference_or_toc(match.start(), text, matched_len):
                return match.start()
        
        return matches[0].start() if matches else None
    
    try:
        section_header = None
        section_header_idx = None
        for index, header in enumerate(toc.keys()):
            if header.strip().lower() == section_name.strip().lower():
                section_header = header
                section_header_idx = index
        # Finding Section start
        section_header = section_header if section_header else section_name
        words = list(map(re.escape, section_header.split()))
        upper = r"\s+".join(w.upper() for w in words)
        title = r"\s+".join(
            w.capitalize() if i == 0 or w.lower() not in STOPWORDS else w.lower()
            for i, w in enumerate(words)
        )
        as_is = r"\s+".join(w for w in words)

        upper_parts = upper.split(r"\s+")
        title_parts = title.split(r"\s+")
        as_is_parts = as_is.split(r"\s+")

        # Use flexible pattern with better whitespace handling
        upper_pattern = sep.join(upper_parts)
        title_pattern = sep.join(title_parts)
        as_is_pattern = sep.join(as_is_parts)
        
        # Try to find matches with all three patterns
        # Find first real header match (not cross-reference or TOC entry)
        start_index = None
        upper_header_matches = list(re.finditer(upper_pattern, content))
        if upper_header_matches:
            start_index = find_first_real_header_match(upper_header_matches, content, section_name)
        
        if start_index is None:
            title_header_matches = list(re.finditer(title_pattern, content))
            if title_header_matches:
                start_index = find_first_real_header_match(title_header_matches, content, section_name)
        
        if start_index is None:
            as_is_header_matches = list(re.finditer(as_is_pattern, content))
            if as_is_header_matches:
                start_index = find_first_real_header_match(as_is_header_matches, content, section_name)
        
        if start_index is None:
            if verbose:
                print(f"[WARN] Fallback could not find section start for '{section_name}'")
            return ""

        # Finding section end which is next MAJOR section header (skipping subsections)
        # This fixes the bug where Risk Factors subsections were being treated as section boundaries
        next_section_header = _find_next_major_section(toc, section_header, section_header_idx)
        
        if verbose and next_section_header:
            # Show what we skipped
            toc_keys = list(toc.keys())
            skipped = []
            for i in range(section_header_idx + 1, len(toc_keys)):
                if toc_keys[i] == next_section_header:
                    break
                skipped.append(toc_keys[i])
            if skipped:
                print(f"[DEBUG] Skipping {len(skipped)} subsections to reach '{next_section_header}'")
                for s in skipped[:3]:
                    print(f"[DEBUG]   - Skipped: {s[:60]}...")
        
        if not next_section_header:
            # No next section, use rest of document
            section_text = content[start_index:]
            cleaned_section_text = re.sub(r"\s*\n\s*|\s{2,}", " ", section_text).strip()
            return cleaned_section_text
        
        # Use flexible pattern for next section that handles optional words
        # Use word boundaries to prevent matching within longer phrases
        # (e.g., "Exchange Offer" shouldn't match "EXCHANGE OFFER PROCEDURES")
        flexible_pattern = _build_flexible_section_pattern(next_section_header, require_word_boundaries=True)
        
        # Search for next section AFTER the start of current section
        # Use first match after start_index, not last match in entire document
        search_content = content[start_index + 50:]  # Skip a bit past section header
        next_header_matches = list(re.finditer(flexible_pattern, search_content))
        
        # Find the first next section match that looks like a real header
        # (not just an inline reference like "the Company reported...")
        end_index = -1
        for m in next_header_matches:
            match_pos_in_search = m.start()
            match_len = m.end() - m.start()
            actual_pos = start_index + 50 + match_pos_in_search
            
            # Check if this looks like a standalone section header
            if is_standalone_header(actual_pos, content, match_len):
                end_index = actual_pos
                break
            
            # Also check if it's NOT a cross-reference or inline text
            if not is_cross_reference_or_toc(actual_pos, content, match_len):
                # Check if it's at the start of a line with minimal text before
                line_start = content.rfind('\n', 0, actual_pos)
                if line_start == -1:
                    line_start = 0
                else:
                    line_start += 1
                text_before = content[line_start:actual_pos]
                
                # For next section boundary: require EITHER
                # 1. Very little text before (just whitespace/indentation) - true standalone header
                # 2. ALL CAPS matched text (like "THE COMPANY" vs "the Company")
                matched_text = content[actual_pos:actual_pos + match_len]
                is_all_caps = matched_text.upper() == matched_text
                
                # Skip if there's significant text before on the same line
                # This filters out inline references like "Although the Company reported..."
                if len(text_before.strip()) > 5 and not is_all_caps:
                    continue
                
                # CRITICAL FIX: For section boundary detection, REQUIRE ALL CAPS
                # even if text_before is empty. This is because wrapped sentences
                # can start a line with "the Company" which is NOT a section header.
                # Real section headers are typically ALL CAPS like "THE COMPANY"
                # or at least Title Case with explicit formatting markers.
                if not is_all_caps:
                    # Not all caps - likely inline reference even at line start
                    # (could be a wrapped sentence continuation)
                    continue
                
                # NEW: Skip if text before is ALSO ALL CAPS - this indicates a compound
                # header like "FACTORS RELATING TO THE COMPANY" where we matched "THE COMPANY"
                text_before_stripped = text_before.strip()
                if text_before_stripped and text_before_stripped.upper() == text_before_stripped:
                    # Text before is also ALL CAPS - skip this match as it's part of a compound header
                    continue
                
                end_index = actual_pos
                break
        
        section_text = content[start_index:end_index] if end_index != -1 else content[start_index:]
        cleaned_section_text = re.sub(r"\s*\n\s*|\s{2,}", " ", section_text).strip()
        return cleaned_section_text
    except Exception as e:
        if verbose:
            print(
                f"[WARN] Fallback extractor failed to extract section '{section_name}': {e}"
            )
        return ""
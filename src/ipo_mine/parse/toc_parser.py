from bs4 import BeautifulSoup, Tag
from collections import OrderedDict # not necessary after we fix the parser
import glob
import re
from typing import Optional, Callable, Dict, List, Tuple
from utils.config import print_config, RAW_DIR, OUTPUT_DIR, PARSED_DIR, set_data_root

# Page tokens at end of TOC rows:
#   - Arabic numbers: "93"
#   - Roman numerals: "ii", "xii"
#   - Prefixed schedules: "F-1", "S-1" (allow whitespace around dash)
PAGE_RE   = re.compile(r"^(?:[A-Z]\s*-\s*\d+|\d+|[ivxlcdm]+)$", re.IGNORECASE)
F_PAGE_RE = re.compile(r"^[A-Z]\s*-\s*\d+$", re.IGNORECASE)  # used only to *allow* non-numeric pages like F-1

# Normalize Unicode dashes so "F–1" == "F-1"
DASH_TRANSLATION = str.maketrans({
    "\u2010": "-",  # hyphen (‐)
    "\u2011": "-",  # non-breaking hyphen
    "\u2012": "-",  # figure dash (‒)
    "\u2013": "-",  # en dash (–)
    "\u2014": "-",  # em dash (—)
    "\u2015": "-",  # horizontal bar (―)
    "\u2212": "-",  # minus sign (−)
})

# Junk guard: EIN often shows up in header tables near the TOC
EIN_RE = re.compile(r"\b\d{2}-\d{7}\b")

# If we’re in the MAIN TOC (not reading body content), a purely numeric page this large is nonsense (e.g., “3600”)
MAX_REASONABLE_PAGE = 1000

def _norm(s: str) -> str:
    """Whitespace + dash normalization and removal of trailing dot leaders."""
    s = s or ""
    # Strip invisible Unicode characters: NBSP, zero-width space, zero-width non-joiner, zero-width joiner
    s = s.replace("\xa0", " ").replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
    s = s.translate(DASH_TRANSLATION)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s*\.{2,}\s*$", "", s)
    return s

def _looks_numeric_heavy(s: str) -> bool:
    """Skip rows that are mostly numbers (common in nearby data tables)."""
    digits = sum(ch.isdigit() for ch in s)
    return digits >= 10 or (len(s) >= 25 and digits / max(1, len(s)) > 0.25)

def _row_to_title_page(tr: Tag) -> tuple[str | None, str | None]:
    """
    Given a <tr>, return (title, page) if *any* cell (preferring right-most) looks like a page token,
    else (None, None). Skips empty cells (containing only &nbsp; or whitespace).
    """
    tds = tr.find_all("td")
    if len(tds) < 2:
        return None, None

    # Page is taken from the right-most page-like cell (skipping empty cells)
    page_txt, page_td = None, None
    for td in reversed(tds):
        txt = _norm(td.get_text("", strip=True))
        # Skip empty cells (common nbsp spacers)
        if not txt:
            continue
        if PAGE_RE.match(txt):
            page_txt, page_td = txt, td
            break
    if page_txt is None:
        return None, None

    # Title is taken from the remaining cells (prefer <a> text), skipping empty cells
    left_cells = [td for td in tds if td is not page_td]
    title = None
    for td in left_cells:
        a = td.find("a")
        if a:
            atxt = _norm(a.get_text("", strip=True))
            if atxt:
                title = atxt
                break
    if not title:
        # Concatenate text from non-empty cells
        parts = []
        for td in left_cells:
            txt = _norm(td.get_text("", strip=True))
            if txt:
                parts.append(txt)
        title = " ".join(parts)

    # Basic quality filters
    if not title or len(title) < 3 or title.isdigit():
        return None, None
    if "table of contents" in title.lower():
        return None, None
    if EIN_RE.search(title):
        return None, None

    # Main-TOC sanity on numeric pages (but allow schedules like F-1, S-1)
    if page_txt.isdigit() and int(page_txt) > MAX_REASONABLE_PAGE:
        return None, None

    if _looks_numeric_heavy(title):
        return None, None

    return title, page_txt

def _is_toc_table(tbl: Tag) -> bool:
    """
    Heuristic for a *TOC-looking* table:
      - Has a 'Page' header somewhere, OR
      - >= 5 rows whose right-most non-empty cell is a page token.
    """
    try:
        header_hit = any("page" in _norm(x.get_text()).lower() for x in tbl.find_all(["td", "th"]))
    except Exception:
        header_hit = False

    hits = 0
    for tr in tbl.find_all("tr", recursive=True)[:160]:
        tds = tr.find_all("td")
        if len(tds) < 2:
            continue
        # Find right-most non-empty cell
        for td in reversed(tds):
            last = _norm(td.get_text("", strip=True))
            if last:  # Skip empty cells
                if PAGE_RE.match(last):
                    hits += 1
                break
        if hits >= 5:
            break

    return header_hit or hits >= 5

def _contains_page_token(text: str) -> bool:
    """
    Return True if `text` exposes a page token in a typical TOC format.

    Accepted forms:
      - Entire text is a page token (e.g., "17", "ii", "F-1")
      - Last token is a page token (e.g., "RISK FACTORS 17")
      - Last token wrapped in parentheses (e.g., "(ii)")
    """
    s = _norm(text or "")
    if not s:
        return False

    # Case 1: whole string is a page token
    if PAGE_RE.fullmatch(s):
        return True

    parts = s.split()
    if not parts:
        return False

    last = parts[-1]

    # Case 2: last token is a page token
    if PAGE_RE.fullmatch(last):
        return True

    # Case 3: parenthesized page token (common for roman numerals)
    if last.startswith("(") and last.endswith(")"):
        inner = last[1:-1].strip()
        if PAGE_RE.fullmatch(inner):
            return True

    return False

def _has_link_only_toc(soup: BeautifulSoup) -> bool:
    """
    True only when:
      - a real TOC header exists (not a nav link), and
      - immediately after that header we see multiple TOC-like links, and
      - within that bounded TOC block we see *no* page tokens.

    Important: the scan is intentionally bounded to avoid false negatives from
    standalone PDF page markers like "2", "3" that appear after the TOC block.
    """
    def _is_toc_header(tag: Tag) -> bool:
        if not isinstance(tag, Tag):
            return False
        if tag.name not in {"p","b","div","h1","h2","h3","h4","h5","h6","font","span","td"}:
            return False
        txt = _norm(tag.get_text(" ", strip=True)).lower()
        return "table of contents" in txt

    def _looks_like_body_paragraph(t: str) -> bool:
        # Heuristic: body paragraphs are long and not “header-like”
        if len(t) < 140:
            return False
        letters = sum(ch.isalpha() for ch in t)
        if letters < 60:
            return False
        # TOC headers are often uppercase; paragraphs typically are not
        upper = sum(ch.isupper() for ch in t)
        return upper / max(1, letters) < 0.70

    # 1) Find a non-navigation TOC header
    hdr = None
    for cand in soup.find_all(_is_toc_header):
        if _is_toc_nav_link(cand):
            continue
        if cand.parent and isinstance(cand.parent, Tag) and _is_toc_nav_link(cand.parent):
            continue
        if len(cand.get_text(" ", strip=True)) > 100:
            continue
        hdr = cand
        break
    if not hdr:
        return False

    # 2) Scan forward but STOP once we hit the first real paragraph after the TOC list.
    # Collect link texts and track whether any page token appears inside the TOC block.
    link_texts = []
    saw_page_token = False

    for node in hdr.find_all_next(["a", "td", "th", "p", "div", "span", "font", "br"], limit=400):
        t = _norm(node.get_text(" ", strip=True))

        # If we’ve already seen a real TOC list (>=5 links) and we hit a body paragraph,
        # we’re past the TOC block. Stop BEFORE we accidentally see stray "2", "3" markers.
        if link_texts and len(link_texts) >= 5 and t and _looks_like_body_paragraph(t):
            break

        # Track TOC-like links
        if node.name == "a":
            lt = _norm(node.get_text(" ", strip=True))
            if lt and not EXCLUDE_RE.match(lt) and len(lt) <= 120:
                link_texts.append(lt)

        # Track page tokens *inside the TOC block*
        # Only consider page tokens if they appear in a context that could be TOC-related.
        # (This avoids counting random standalone "2"/"3" later, because we stop early.)
        if t and _contains_page_token(t):
            saw_page_token = True

    # Need enough links to call it a TOC list (avoid nav menus)
    if len(link_texts) < 5:
        return False

    # Links but no page tokens in the bounded TOC block => link-only TOC
    return not saw_page_token


def _is_toc_nav_link(tag: Tag) -> bool:
    """
    Check if a tag is just a navigation link to TOC (not the actual TOC header).
    Navigation links typically look like: <a href="#toc">Table of Contents</a>
    """
    if tag.name == "a" and tag.get("href", "").startswith("#"):
        return True
    # Check if the TOC text is inside a link with href="#..."
    a_tag = tag.find("a", href=True)
    if a_tag and a_tag.get("href", "").startswith("#"):
        text = tag.get_text("", strip=True).lower()
        # If the entire text is basically just "table of contents", it's a nav link
        if text.replace("table of contents", "").strip() == "":
            return True
    return False

def _find_toc_tables_block(soup: BeautifulSoup) -> list[Tag]:
    """
    Find the TOC heading, then collect *consecutive* tables that look like a TOC.
    Stop at the first non-TOC table. This approximates "only the TOC page(s)".
    """
    def _normalize_toc_text(s: str) -> str:
        """Normalize whitespace/newlines for 'table of contents' matching."""
        return " ".join(s.lower().split())
    
    # 1) Prefer the explicit heading (skip navigation links and large containers)
    hdr = None
    for candidate in soup.find_all(
        lambda tag: isinstance(tag, Tag)
        and tag.name in {"p", "b", "div", "h1", "h2", "h3", "h4", "h5", "h6", "font", "span", "td"}
        and "table of contents" in _normalize_toc_text(tag.get_text("", strip=True))
    ):
        # Skip navigation links
        if _is_toc_nav_link(candidate):
            continue
        # Skip if parent is a navigation link
        if candidate.parent and _is_toc_nav_link(candidate.parent):
            continue
        # Skip large container elements - actual TOC headers are typically short
        text = candidate.get_text("", strip=True)
        if len(text) > 100:
            continue
        hdr = candidate
        break
    tables: list[Tag] = []
    if hdr:
        node = hdr
        # Walk forward in DOM order grabbing tables until a non-TOC table appears.
        while True:
            node = node.find_next(["table", "div", "p", "section"])
            if node is None:
                break
            if isinstance(node, Tag) and node.name == "table":
                if _is_toc_table(node):
                    tables.append(node)
                    # keep going—some filers split TOC across 2 tables
                    continue
                # first non-TOC table ⇒ TOC block ends
                break
            # If we hit a big content block before any table appears (rare), bail out.
            if len(tables) == 0 and isinstance(node, Tag) and node.name in {"section", "div"} and node.find("table") is None:
                # keep scanning; not fatal—just don’t return early
                pass

    # 2) Fallback: no heading found — pick the first TOC-looking table, then its immediate TOC-looking siblings
    if not tables:
        first = None
        for t in soup.find_all("table"):
            if _is_toc_table(t):
                first = t
                break
        if first:
            tables.append(first)
            sib = first.find_next_sibling("table")
            while sib and _is_toc_table(sib):
                tables.append(sib)
                sib = sib.find_next_sibling("table")

    return tables

# Known S-1 section names for validation (case-insensitive matching)
KNOWN_S1_SECTIONS = {
    "prospectus summary", "summary", "risk factors", "risks", 
    "use of proceeds", "dividend policy", "capitalization", "dilution",
    "management", "management's discussion and analysis", "mdna", "md&a",
    "business", "description of capital stock", "description of securities",
    "principal stockholders", "principal and selling stockholders",
    "selling stockholders", "certain relationships", "related party transactions",
    "underwriting", "legal matters", "experts", "where you can find more information",
    "incorporation by reference", "index to financial statements",
    "index to consolidated financial statements", "financial statements",
    "executive compensation", "security ownership", "shares eligible for future sale",
    "material u.s. federal income tax considerations", "tax considerations",
    "plan of distribution", "description of certain indebtedness",
    "market and industry data", "forward-looking statements",
    "special note regarding forward-looking statements", "glossary",
    "the offering", "our company", "about this prospectus",
}

# Patterns to exclude - these look like section headers but are not main TOC sections
EXCLUDE_PATTERNS = [
    r"^(united states|securities and exchange commission|sec|washington)",
    r"^(preliminary|subject to completion)",
    r"^(per share|total|actual|pro forma)",
    r"^(year|ended|as of|december|january|february|march|april|may|june|july|august|september|october|november)",
    r"^(number|shares|voting|stock|price|common)\b",
    r"^(table|page|part|item)\b",
    r"^\d",  # starts with number
    r"^(before|after)\s+(the\s+)?offering",
    r"^\(?(inception|to)\)?\b",  # (inception) to
    r"^(percentage|ownership|option to|additional|paid|description of exhibit)",
    r"^(signatures?|power of attorney|date)\b",
    r"^(exhibit|schedule)\s",
    r"^information not required",
]
EXCLUDE_RE = re.compile("|".join(EXCLUDE_PATTERNS), re.IGNORECASE)

# Patterns to match "Risk Factors" in various forms
RISK_FACTORS_RE = re.compile(
    r"^\s*(?:RISK\s+FACTORS?|Risk\s+Factors?)\s*$",
    re.IGNORECASE
)


def _extract_style_dict(style_str: str) -> Dict[str, str]:
    """Parse an inline CSS style string into a dictionary."""
    if not style_str:
        return {}
    styles = {}
    for item in style_str.split(";"):
        item = item.strip()
        if ":" in item:
            key, val = item.split(":", 1)
            styles[key.strip().lower()] = val.strip().lower()
    return styles


def _get_element_style_signature(tag: Tag) -> Dict[str, str]:
    """
    Extract a 'style signature' from an element, including inherited styles.
    Focuses on properties that typically distinguish section headers.
    """
    sig = {}
    
    # Direct style attribute
    style_str = tag.get("style", "")
    direct_styles = _extract_style_dict(style_str)
    
    # Key styling properties for section headers
    key_props = [
        "font-weight", "font-size", "font-family", "text-transform",
        "text-align", "font-style", "text-decoration"
    ]
    
    for prop in key_props:
        if prop in direct_styles:
            sig[prop] = direct_styles[prop]
    
    # Check parent chain for inherited styles (up to 3 levels)
    parent = tag.parent
    for _ in range(3):
        if parent is None or not isinstance(parent, Tag):
            break
        parent_style = _extract_style_dict(parent.get("style", ""))
        for prop in key_props:
            if prop not in sig and prop in parent_style:
                sig[prop] = parent_style[prop]
        parent = parent.parent
    
    # Check for bold/style via tag name
    if tag.name == "b" or tag.name == "strong":
        sig.setdefault("font-weight", "bold")
    if tag.name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
        sig.setdefault("font-weight", "bold")
    
    # Check for font tag with face attribute (older HTML)
    if tag.name == "font":
        face = tag.get("face", "")
        if face:
            sig.setdefault("font-family", face.lower())
    
    return sig


def _styles_match(sig1: Dict[str, str], sig2: Dict[str, str], tolerance: int = 1) -> bool:
    """
    Check if two style signatures are similar enough.
    tolerance = number of differences allowed.
    """
    # Must match on key distinguishing properties
    critical = ["text-transform", "font-weight"]
    for prop in critical:
        if sig1.get(prop) != sig2.get(prop):
            # text-transform: uppercase is critical for many S-1s
            if prop == "text-transform":
                return False
    
    # Count differences in other properties
    all_props = set(sig1.keys()) | set(sig2.keys())
    diffs = sum(1 for p in all_props if sig1.get(p) != sig2.get(p))
    
    return diffs <= tolerance


def _find_risk_factors_header(soup: BeautifulSoup) -> Optional[Tag]:
    """
    Find the "RISK FACTORS" section header element.
    Returns the most specific element containing just "Risk Factors" text.
    """
    candidates = []
    
    # Search in common header containers
    for tag in soup.find_all(["div", "p", "span", "font", "b", "strong", "h1", "h2", "h3", "h4", "td"]):
        text = tag.get_text("", strip=True)
        
        # Must be a standalone "Risk Factors" header, not a reference
        if not RISK_FACTORS_RE.match(text):
            continue
        
        # Skip if it's inside an anchor (likely a TOC entry or nav link)
        if tag.find_parent("a"):
            continue
        
        # Skip if it contains an anchor with href (reference, not header)
        if tag.find("a", href=True):
            continue
        
        # Check for header-like styling
        style_sig = _get_element_style_signature(tag)
        
        # Prefer elements with explicit styling
        score = 0
        if style_sig.get("text-transform") == "uppercase":
            score += 10
        if style_sig.get("font-weight") == "bold":
            score += 5
        if style_sig.get("text-align") == "center":
            score += 3
        if tag.name in {"h1", "h2", "h3", "h4"}:
            score += 5
        
        # Penalize if text is too long (probably not a header)
        if len(text) > 30:
            score -= 10
        
        candidates.append((tag, score, style_sig))
    
    if not candidates:
        return None
    
    # Sort by score (descending) and return the best match
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


def _find_sections_by_style(soup: BeautifulSoup, reference_tag: Tag) -> List[Tuple[str, Tag]]:
    """
    Find all elements with similar styling to the reference tag (Risk Factors header).
    Returns list of (section_title, tag) tuples.
    """
    ref_style = _get_element_style_signature(reference_tag)
    sections = []
    seen_titles = set()
    
    # Search for similarly styled elements
    for tag in soup.find_all(["div", "p", "span", "font", "b", "strong", "h1", "h2", "h3", "h4", "td"]):
        text = _norm(tag.get_text("", strip=True))
        
        # Skip empty or too-long text
        if not text or len(text) < 3 or len(text) > 80:
            continue
        
        # Skip numeric-heavy text
        if _looks_numeric_heavy(text):
            continue
        
        # Skip TOC references
        if "table of contents" in text.lower():
            continue
        
        # Skip if inside anchor (navigation links)
        if tag.find_parent("a"):
            continue
        
        # Skip excluded patterns (dates, numbers, table headers, etc.)
        if EXCLUDE_RE.match(text):
            continue
        
        # Check style similarity
        tag_style = _get_element_style_signature(tag)
        if not _styles_match(ref_style, tag_style):
            continue
        
        # Normalize title for dedup
        title_normalized = text.lower().strip()
        if title_normalized in seen_titles:
            continue
        
        # Must match known S-1 section OR have header-like characteristics
        is_known = any(known in title_normalized or title_normalized in known 
                      for known in KNOWN_S1_SECTIONS)
        
        # For non-known sections, be more strict
        if not is_known:
            # Must be short (headers are typically 1-6 words)
            if len(text.split()) > 6:
                continue
            # Skip single words that aren't known sections
            if len(text.split()) == 1 and len(text) < 10:
                continue
        
        seen_titles.add(title_normalized)
        sections.append((text, tag))
    
    return sections


def _estimate_section_order(sections: List[Tuple[str, Tag]], soup: BeautifulSoup) -> List[Tuple[str, str]]:
    """
    Order sections by their position in the document and assign placeholder page numbers.
    Returns list of (title, page) tuples.
    """
    # Use sourceline if available (much faster than searching descendants)
    def get_position(tag: Tag) -> int:
        if hasattr(tag, 'sourceline') and tag.sourceline is not None:
            return tag.sourceline
        return 0
    
    # Sort by document position
    sections_with_pos = [(title, tag, get_position(tag)) for title, tag in sections]
    sections_with_pos.sort(key=lambda x: x[2])
    
    # Assign sequential placeholder page numbers (we don't have actual pages)
    # Use ~ prefix to indicate estimated order
    result = []
    for i, (title, tag, _) in enumerate(sections_with_pos, start=1):
        result.append((title, f"{i}"))
    
    return result


def parse_toc_from_risk_factors_style(raw_content: str) -> OrderedDict:
    """
    Fallback TOC parser that uses the "Risk Factors" section header as a style reference
    to find other section headers in the document.
    
    This is used when the normal TOC parsing fails. Since S-1 filings are required
    to have a "Risk Factors" section, we can use its styling as a template to
    identify other similarly-styled section headers.
    
    Returns:
        OrderedDict {section → page}
        since actual page numbers aren't available from the content.
    """
    soup = BeautifulSoup(raw_content, 'html.parser')
    
    # Step 1: Find the Risk Factors header
    risk_factors_tag = _find_risk_factors_header(soup)
    if not risk_factors_tag:
        return OrderedDict()
    
    # Step 2: Find all sections with similar styling
    sections = _find_sections_by_style(soup, risk_factors_tag)
    if not sections:
        return OrderedDict()
    
    # Step 3: Order sections and create result
    ordered_sections = _estimate_section_order(sections, soup)
    
    return OrderedDict(ordered_sections)


def is_fallback_toc(toc: dict) -> bool:
    """
    Check if a TOC was generated using the fallback (Risk Factors style) parser.
    
    Fallback TOCs have page numbers prefixed with '~' to indicate they are
    estimated order positions rather than actual page numbers.
    
    Args:
        toc: The parsed TOC dictionary
        
    Returns:
        True if the TOC was generated via fallback parsing, False otherwise.
    """
    if not toc:
        return False
    # Check if any page number starts with ~
    return any(str(v).startswith('') for v in toc.values())


def parse_toc_2015_2025(
    raw_content: str,
    return_dict: bool = True,  # kept for compatibility; always returns OrderedDict
    use_fallback: bool = True,
):
    """
    Parse the TABLE OF CONTENTS across *all* TOC tables that appear directly after
    the TOC heading. We stop at the first table that does *not* look like a TOC.
    No hard-coded section names. No FS-specific logic. No page-count guesses.

    Returns:
        OrderedDict {section → page} in document order.
        Empty OrderedDict indicates a short / partial / omitted filing.
    """

    MIN_STRIPPED_TEXT_LEN = 11000

    # --- Structural soup (DO NOT MUTATE) ---
    soup = BeautifulSoup(raw_content, "html.parser")

    # --- Narrative-text soup (SAFE TO MUTATE) ---
    text_soup = BeautifulSoup(raw_content, "html.parser")
    for tag in text_soup(["table", "script", "style", "noscript"]):
        tag.decompose()

    text = text_soup.get_text(" ", strip=True)
    text_lower = text.lower()

    # Only inspect the beginning of the filing to avoid false positives
    top = text_lower[:8000]

    OMISSION_PHRASES = (
        "therefore been omitted",
        "has been omitted",
        "have been omitted",
        "solely to file certain exhibits"
    )

    has_omission = any(p in top for p in OMISSION_PHRASES)

    # Early exit 1: genuinely short filings (after removing tables)
    if len(text) < MIN_STRIPPED_TEXT_LEN:
        return OrderedDict()

    # Early exit 2: explicit omission of substantive content
    if has_omission:
        return OrderedDict()
    
    # Short-circuit: some modern SEC filings include a "Table of Contents" that is
    # implemented purely as anchor links (no page numbers exposed in the HTML).
    # For this parser, a TOC without page numbers is not considered a valid TOC,
    # since we require a section → page mapping and do not infer or hallucinate pages.
    #
    # This check must occur before table-based parsing and before fallback logic
    # to avoid incorrectly synthesizing a TOC from body headers or styles.
    if _has_link_only_toc(soup):
        return OrderedDict()

    # --- Normal TOC parsing (uses untouched soup) ---
    toc_tables = _find_toc_tables_block(soup)
    if not toc_tables:
        # As a last resort, scan the whole doc (rare)
        toc_tables = soup.find_all("table")

    rows: list[tuple[str, str]] = []
    for tbl in toc_tables:
        for tr in tbl.find_all("tr", recursive=True):
            title, page = _row_to_title_page(tr)
            if title and page:
                rows.append((title, page))

    # De-dup while keeping first occurrence
    seen, ordered = set(), []
    for t, p in rows:
        if t not in seen:
            seen.add(t)
            ordered.append((t, p))

    # Normal success
    if len(ordered) >= 4:
        return OrderedDict(ordered)

    # Fallback parser
    if use_fallback:
        fallback_result = parse_toc_from_risk_factors_style(raw_content)
        if fallback_result and len(fallback_result) >= 4:
            return fallback_result

    # Normalize tiny / missing TOCs to empty
    return OrderedDict()

def parse_toc_plain_text(content):
    left_last_numeric_page, right_last_numeric_page = None, None
    def add_entry_tc(title: str, page: str, curr_entries: list, on_left_side: bool) -> bool:
        """
        Append (title, page) to curr_entries.
        Return True if we should STOP (numeric page backslides too far or invalid).
        Buffer-free: `title` should already be fully assembled by the caller.
        """
        nonlocal left_last_numeric_page, right_last_numeric_page

        last_numeric_page = left_last_numeric_page if on_left_side else right_last_numeric_page

        kind = page_kind(page)
        if not kind:
            return False  # ignore invalid page

        title = title.strip()
        if not title:
            return False

        if len(title) >= header_limit:
            return True

        if kind == "num":
            try:
                n = int(page)
            except ValueError:
                return False

            if (last_numeric_page is not None and n < last_numeric_page) or n >= 500:
                return True
            last_numeric_page = max(last_numeric_page or n, n)

        curr_entries.append((title, page))

        if on_left_side:
            left_last_numeric_page = last_numeric_page
        else:
            right_last_numeric_page = last_numeric_page

        return False

    def parse_two_column_toc_lines(lines, entry_re, PAGE, ignore_terms):
        """
        Fully parse a two-column TOC region, including:
        - two entries on one physical line (left complete + right complete)
        - right-column wrap where right fragment is completed either by:
            (A) a standalone indented "... page" line, OR
            (B) the right side of the NEXT two-column line  <-- your screenshot case
        - left-column wrap where left is incomplete but right is complete
        - lone left-only entries
        Returns: left_entries + right_entries
        """

        left_entries, right_entries = [], []

        left_buffer = []   # left fragments w/o page
        right_buffer = []  # right fragments w/o page

        def is_ignored(s: str) -> bool:
            return any(term.lower() in s.lower() for term in ignore_terms)

        two_col_re = re.compile(
            rf'^(?P<l_title>.+?)\s*(?:[.\s]{{2,}})\s*(?P<l_page>{PAGE})\s{{2,}}(?P<r_part>.+)$',
            re.I
        )

        left_frag_right_re = re.compile(r'^(?P<l_frag>.+?)\s{2,}(?P<r_part>\S.*)$')

        for raw in lines:
            s = raw.strip()
            if not s or is_ignored(s):
                continue

            m_entry = entry_re.match(s)

            # --- Case B FIRST: two-column line where LEFT is complete ---
            tc = two_col_re.match(s)
            if tc:
                l_title = tc.group("l_title").strip()
                l_page  = tc.group("l_page").strip()
                r_part  = tc.group("r_part").strip()

                # complete wrapped left title if needed
                if left_buffer:
                    l_title = " ".join(left_buffer + [l_title]).strip()
                    left_buffer.clear()

                if add_entry_tc(l_title, l_page, left_entries, True):
                    break

                # parse right side
                mr = entry_re.match(r_part)
                if mr:
                    r_title, r_page = mr.group(1).strip(), mr.group(2).strip()

                    # ✅ KEY FIX: if right title was buffered from previous line,
                    # treat this right entry as the continuation and prepend it.
                    if right_buffer:
                        r_title = " ".join(right_buffer + [r_title]).strip()
                        right_buffer.clear()

                    if add_entry_tc(r_title, r_page, right_entries, False):
                        break
                else:
                    # right side is a wrapped fragment (no page yet)
                    right_buffer = [r_part] if r_part else []
                continue

            # --- Case A: wrapped RIGHT title completed by standalone indented "... page" line ---
            if right_buffer and raw.startswith((" ", "\t")) and m_entry:
                frag, page = m_entry.group(1).strip(), m_entry.group(2).strip()
                full_title = " ".join(right_buffer + [frag]).strip()
                right_buffer.clear()

                if add_entry_tc(full_title, page, right_entries, False):
                    break
                continue

            # --- Case C: left fragment + right complete (screenshot #1 you showed earlier) ---
            lfr = left_frag_right_re.match(s)
            if lfr:
                l_frag = lfr.group("l_frag").strip()
                r_part = lfr.group("r_part").strip()

                mr = entry_re.match(r_part)
                if mr and not entry_re.match(l_frag):
                    if l_frag:
                        left_buffer.append(l_frag)

                    r_title, r_page = mr.group(1).strip(), mr.group(2).strip()
                    if add_entry_tc(r_title, r_page, right_entries, False):
                        break
                    continue

            # --- Case D: single entry line (left-only) ---
            if m_entry:
                left, page = m_entry.group(1).strip(), m_entry.group(2).strip()
                if left_buffer:
                    left = " ".join(left_buffer + [left]).strip()
                    left_buffer.clear()

                if add_entry_tc(left, page, left_entries, True):
                    break
                continue

            # otherwise: treat as left-wrap fragment
            left_buffer.append(s)
            if len(left_buffer) > 20:
                break

        return left_entries + right_entries



    # --- Locate TOC region ---
    table_idx = content.find('TABLE OF CONTENTS')
    table_idx = table_idx if table_idx != -1 else content.find('Table of Contents')
    table_idx = table_idx if table_idx != - 1 else content.find('Table of contents')
    if table_idx == -1:
        print('[ERROR]: Could not locate table of contents')

    toc = content[table_idx:]
    c_idx = toc.find('<C>')
    if c_idx != -1 and c_idx < 500:
        toc = toc[c_idx:].strip()

    lines = toc.splitlines()[1:]
    header_limit = 150
    entries, buffer = [], []
    two_column = False

    # --- Patterns ---
    ROMAN = r'[ivxlcdmIVXLCDM()]{1,10}'
    PAGE  = rf'([A-Z]-?(?:\d{{1,3}}|{ROMAN})|\d{{1,3}}|{ROMAN})'
    entry_re = re.compile(rf'^(.*?)\s*(?:[.\s]{{2,}})\s*{PAGE}(?:\s+.*)?$', re.I)
    big_gap_re      = re.compile(r'^(.*?)\s{8,}(\S.*)$')
    right_head_page = re.compile(rf'^{PAGE}\b', re.I)
    num_re          = re.compile(r'^\d{1,3}$')
    alpha_re        = re.compile(r'^[A-Z]-?\d{1,3}$', re.I)
    roman_re        = re.compile(ROMAN)
    two_col_split_re = re.compile(
        rf'^(.+?)(?:[.\s]{{2,}})\s*{PAGE}\s{{2,}}(.+)$',
        re.I
    )

    # Ignore terms for multi-page TOCs
    ignore_terms = ("<PAGE>", "<TABLE>", "<CAPTION>", "<C>", "</PAGE>", "</TABLE>", "PAGE", "NO.", "----")

    def page_kind(p: str):
        if num_re.match(p):
            try:
                return 'num' if int(p) < 500 else None
            except ValueError:
                return None
        if alpha_re.match(p):
            return 'alpha' 
        return 'roman' if roman_re.match(p) else None

    # --- Monotonic stop state ---
    last_numeric_page = None

    def add_entry(left: str, page: str, curr_entries: list) -> bool:
        """Append (title, page). Return True if we should STOP (numeric page backslides too far)."""
        nonlocal last_numeric_page
        kind = page_kind(page)
        if not kind:
            return False  # ignore invalid page
        title = (" ".join(buffer + [left])).strip() if buffer else left.strip()
        if len(title) >= header_limit:
            return True
        if not title:
            return False
        if kind == 'num':
            n = int(page)
            if (last_numeric_page is not None and n < last_numeric_page) or n >= 500:
                return True
            last_numeric_page = max(last_numeric_page or n, n)
        curr_entries.append((title, page))
        buffer.clear()
        return False

    # --- Parse ---
    for raw in lines:
        # Subsection entry check
        if entry_re.search(raw) and raw.startswith((" ", "\t")) and buffer == []:
            continue
        s = raw.strip()
        if not s:
            continue
        # Ignoring gaps between pages
        if any(term.lower() in s.lower() for term in ignore_terms):
            continue
        # Check for two-column
        tc = two_col_split_re.match(s)
        if tc:
            two_column = True
            break

        # 1) Normal entry (dots or spaces before page)
        m = entry_re.match(s)
        if m:
            left, page = m.group(1).strip(), m.group(2).strip()
            if add_entry(left, page, entries):
                break
            continue

        # 2) Big-gap two-column layout
        mg = big_gap_re.match(s)
        if mg:
            left, right = mg.groups()
            left, right = left.rstrip(), right.strip()
            mh = right_head_page.match(right)
            if mh:
                page = mh.group(1)
                if add_entry(left, page, entries):
                    break
                continue
            # otherwise it’s a wrapped title fragment
            if left.strip():
                buffer.append(left.strip())
            continue
        # 3) Plain wrapped title continuation
        buffer.append(s)
        if len(buffer) > 20:
            break
    if two_column:
        entries = parse_two_column_toc_lines(lines, entry_re, PAGE, ignore_terms)
        return dict(entries)
    # Final: if buffer itself is "title + page", parse it under the same rule
    if buffer:
        candidate = " ".join(buffer).strip()
        m_last = entry_re.match(candidate)
        if m_last:
            left, page = m_last.group(1).strip(), m_last.group(2).strip()
            if len(left) <= header_limit:
                # apply same monotonic stop check
                kind = page_kind(page)
                if kind == 'num':
                    n = int(page)
                    print(last_numeric_page)
                    if last_numeric_page is None or n < last_numeric_page:
                        entries.append((left, page))
                elif kind == 'alpha':
                    entries.append((left, page))
            
    return dict(entries)


def parse_toc_1997(raw_content: str, use_fallback: bool = True):
    """Parse TOC from HTML for 1997-era filings."""
    toc = {}
    soup = BeautifulSoup(raw_content, 'html.parser')

    toc_header = soup.find(lambda tag: tag.name in ['p', 'b', 'h1', 'h2', 'h3'] and 'table of contents' in tag.get_text(strip=True).lower())
    if toc_header:
        toc_section = toc_header.find_next(lambda tag: tag.name in ['pre', 'a', 'ul', 'ol'])
        if toc_section:
            for link in toc_section.find_all('a'):
                text = link.get_text(strip=True)
                next_sibling = link.next_sibling
                page_num = None
                while next_sibling and not page_num:
                    sibling_text = getattr(next_sibling, 'text', str(next_sibling)).strip()
                    num_match = re.search(r'^(\d+|[ivxlcdm]+)$', sibling_text, re.IGNORECASE)
                    if num_match:
                        page_num = num_match.group(1)
                    next_sibling = getattr(next_sibling, 'next_sibling', None)

                if text and page_num:
                    toc[text] = page_num

    # If we found reasonable results, return them
    if len(toc) >= 3:
        return toc
    
    # Fallback: Use Risk Factors style-based parsing
    if use_fallback:
        fallback_result = parse_toc_from_risk_factors_style(raw_content)
        if fallback_result and len(fallback_result) >= 3:
            return dict(fallback_result)

    return toc

def _attempt_parse_table_2004(toc_table) -> dict:
    toc = {}
    rows = toc_table.find_all('tr')
    for row in rows:
        cols = row.find_all(['td', 'th'])
        if len(cols) >= 2:
            item_text = ' '.join(cols[0].stripped_strings).strip()
            link_in_first_col = cols[0].find('a')
            if link_in_first_col:
                item_text = link_in_first_col.get_text(strip=True)
            
            page_num = None
            for col in cols:
                col_text = ' '.join(col.stripped_strings).strip()
                page_match = re.search(r'^(\d+|[ivxlcdm]+|[A-Z]-\d+)$', col_text, re.IGNORECASE)
                if page_match:
                    page_num = page_match.group(1)
                    break
            if not page_num:
                page_text = ' '.join(cols[-1].stripped_strings).strip()
                page_match = re.search(r'(\d+|[ivxlcdm]+|[A-Z]-\d+)$', page_text, re.IGNORECASE)
                if page_match:
                    page_num = page_match.group(1)
            if not page_num:
                for col in cols:
                    col_text = ' '.join(col.stripped_strings).strip()
                    page_match = re.search(r'\.{3,}\s*(\d+|[ivxlcdm]+|[A-Z]-\d+)$', col_text, re.IGNORECASE)
                    if page_match:
                        page_num = page_match.group(1)
                        break
            if item_text and page_num and len(item_text) > 2:
                item_text = re.sub(r'\s*[\._]{2,}\s*$', '', item_text).strip()
                if len(item_text) > 2:
                     toc[item_text] = page_num
    return toc

def parse_toc_2004_2012(raw_content: str, use_fallback: bool = True):
    """Parse TOC from HTML for 2004-2012 era filings."""
    soup = BeautifulSoup(raw_content, 'html.parser')

    # Extended keywords for marker detection
    toc_keywords = ['table of contents', 'index', 'contents']
    
    # 1. Search for markers
    toc_markers = soup.find_all(
        lambda tag: (
            tag.name in ['p', 'b', 'div', 'h1', 'h2', 'h3', 'font'] 
            and any(k in tag.get_text(strip=True).lower() for k in toc_keywords)
            and len(tag.get_text(strip=True)) < 50  # Avoid matching long paragraphs
        ) or (
            tag.name == 'a' 
            and tag.get('name') 
            and re.search(r'toc|index', tag['name'], re.IGNORECASE)
        )
    )

    for toc_marker in toc_markers:
        # Strategy 1: Look for tables following the marker
        candidate_tables = toc_marker.find_all_next('table', limit=5)
        for table in candidate_tables:
            toc = _attempt_parse_table_2004(table)
            if toc and len(toc) > 2:
                return toc

        # Strategy 2: Lists (ul, ol, div)
        toc_list = None
        for elem in toc_marker.find_next_siblings(['ul', 'ol', 'div'], limit=10):
            if elem.name in ['ul', 'ol']:
                toc_list = elem
                break
            elif elem.name == 'div' and elem.find_all('a', limit=1):
                toc_list = elem
                break

        if toc_list:
            toc = {}
            for link in toc_list.find_all('a'):
                text = link.get_text(strip=True)
                page_num = None
                current = link
                for _ in range(4):
                    next_node = getattr(current, 'next_sibling', None)
                    if next_node is None: break
                    current = next_node
                    node_text = getattr(next_node, 'text', str(next_node)).strip()
                    num_match = re.search(r'^([ivxlcdm\d]+)$', node_text, re.IGNORECASE)
                    if num_match:
                        page_num = num_match.group(1)
                        break
                    if getattr(next_node, 'name', None) in ['a', 'p', 'div', 'li', 'br']:
                        break

                if text and page_num and len(text) > 2:
                    toc[text] = page_num
            
            if toc and len(toc) > 2:
                return toc

    # 2. Fallback: Scan the first 20 tables in the document for TOC-like structure
    # useful if the header is missing or not matched
    all_tables = soup.find_all('table', limit=20)
    for table in all_tables:
        # Heuristic: check if it looks like a TOC
        # We use the same parsing logic; if it yields valid results, we assume it's a TOC
        toc = _attempt_parse_table_2004(table)
        
        # Stronger validation for fallback:
        # - Must have at least 3 entries
        # - Must have at least one known TOC keyword in keys (e.g. "Risk Factors", "Business")
        if toc and len(toc) > 2:
            keys_lower = [k.lower() for k in toc.keys()]
            common_keys = ['risk factors', 'business', 'use of proceeds', 'management', 'underwriting', 'experts', 'legal matters']
            if any(ck in key for key in keys_lower for ck in common_keys):
                return toc

    # 3. Final fallback: Use Risk Factors style-based parsing
    if use_fallback:
        fallback_result = parse_toc_from_risk_factors_style(raw_content)
        if fallback_result and len(fallback_result) >= 3:
            return dict(fallback_result)

    return {}

def parse_toc_2013_2018(raw_content: str):
    """Parse TOC from HTML for 2013-2018 era filings."""
    return parse_toc_2004_2012(raw_content)

def parse_toc_2019_2024(raw_content: str, use_fallback: bool = True):
    """Parse TOC from HTML for 2019-2024 era filings."""
    toc = {}

    soup = BeautifulSoup(raw_content, 'html.parser')

    toc_anchor = soup.find('a', attrs={'name': re.compile(r'toc', re.IGNORECASE)})
    toc_container = None
    if toc_anchor:
        toc_container = toc_anchor.find_parent(['div', 'section'])

    if not toc_container:
        toc_header = soup.find(lambda tag: tag.name in ['p', 'b', 'div', 'h1', 'h2', 'h3'] and 'table of contents' in tag.get_text(strip=True).lower())
        if toc_header:
            container = toc_header.find_next_sibling(['div', 'section', 'p', 'table'])
            if container:
                 if container.name == 'p' and container.parent and container.parent.name in ['div', 'section']:
                      toc_container = container.parent
                 else:
                      toc_container = container
            elif toc_header.parent and toc_header.parent.name in ['div', 'section']:
                 toc_container = toc_header.parent

    if not toc_container:
         toc_container = soup.find('div', style=re.compile(r'font-family:\s*Times New Roman', re.IGNORECASE))

    if toc_container:
        potential_lines = toc_container.find_all(['p', 'div', 'li', 'tr'])
        if not potential_lines:
             container_text = toc_container.get_text(separator='\n', strip=True)
             lines = container_text.split('\n')
        else:
             lines = [el.get_text("", strip=True) for el in potential_lines]

        for line_text in lines:
            line_strip = line_text.strip()
            if not line_strip: continue

            match = re.match(r"^(.*?)(?:\s*\.{2,})?\s+([ivxlcdm\d]+)$", line_strip, re.IGNORECASE)
            if match:
                section, page = match.groups()
                section = section.strip()
                page = page.strip()
                if len(section) > 3 and not section.isdigit():
                    toc[section] = page

    # If we found reasonable results, return them
    if len(toc) >= 3:
        return toc
    
    # Fallback: Use Risk Factors style-based parsing
    if use_fallback:
        fallback_result = parse_toc_from_risk_factors_style(raw_content)
        if fallback_result and len(fallback_result) >= 3:
            return dict(fallback_result)

    return toc

def get_parser_for_year(year) -> Callable[[str], dict[str, str]]:
    """Return the appropriate parser function for a given year."""
    year = int(year)

    if year >= 2015 and year <= 2025:
        return parse_toc_2015_2025
    
    if year < 2000:
        return parse_toc_1997  # For 1997-1999
    elif year < 2013:
        return parse_toc_2004_2012  # For 2000-2012
    elif year < 2019:
        return parse_toc_2013_2018  # For 2013-2018
    else:
        return parse_toc_2019_2024
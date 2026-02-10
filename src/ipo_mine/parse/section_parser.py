"""
Section extraction utilities for S1 filings.

This module provides functions to parse S1 filings into pages and extract
specific sections based on a table of contents.
"""

from typing import Dict, Tuple
import re
from bs4 import BeautifulSoup

# Module-level constants
ROMAN_RE = re.compile(r"^[ivxlcdm]+$", re.IGNORECASE)
ROMAN_MAP = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
ALPHA_PAGE_RE = re.compile(r"^([A-Z]+)\s*[-\s]?\s*([0-9]+)$", re.IGNORECASE)


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
    p = p.strip()
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
            offset = offset * 26 + (ord(ch) - ord('A') + 1)
        return offset * 1000 + int(number)

    cleaned = p.replace(',', '')
    if cleaned.isdigit():
        return int(cleaned)

    raise ValueError(f"Unsupported page label: '{p}'")


def get_section_page_range(toc: Dict[str, str], section_name: str) -> Tuple[int, int]:
    """
    Get the start and end page numbers for a section from the ToC.
    
    Args:
        toc: Table of contents dictionary {section_name: page_number}
        section_name: Name of the section to find
        
    Returns:
        Tuple of (start_page, end_page) as integers
        
    Raises:
        KeyError: If section not found in ToC
    """
    keys = list(toc.keys())
    idx = None
    
    for i, k in enumerate(keys):
        if k.lower().strip() == section_name.lower().strip():
            idx = i
            break
    
    if idx is None:
        raise KeyError(
            f"Section '{section_name}' not found in the Table of Contents. "
            f"Available sections are: {list(toc.keys())}"
        )
    
    start_page = page_to_int(toc[keys[idx]])
    end_page = page_to_int(toc[keys[idx + 1]]) if idx + 1 < len(keys) else (start_page + 100)
    
    return start_page, end_page


def pages_by_bottom_number(raw_text: str) -> Dict[str, str]:
    """Parse plain text filing into pages based on page numbers at bottom."""
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    tag_re = re.compile(r"(?im)(?:^\s*<PAGE[^>]*>\s*(\d+)?\s*$)|(?:</page>)")
    num_re = re.compile(r"^\s*(\d{1,4})\s*$")
    roman_re = re.compile(r"^\s*([IVXLCDMivxlcdm]{1,10})\s*$")
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
                page_label = m_rom.group(1)
                last_idx = idx
                break
            if m_alpha:
                page_label = m_alpha.group(1)
                last_idx = idx
                break
            break
        
        if not page_label:
            unlabeled_count += 1
            page_label = top_num if top_num is not None else str(10000 * unlabeled_count)
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
    raw_html: str, 
    return_html: bool = False  
) -> Dict[str, str]:
    """Parse HTML filing into pages based on <hr> tags and page numbers."""
    html = raw_html.replace("\r\n", "\n").replace("\r", "\n")
    hr_re = re.compile(
        r"(?is)(?:<p[^>]*page-break-before\s*:\s*always[^>]*>.*?)?<hr\b[^>]*>",
        re.IGNORECASE
    )
    parts = hr_re.split(html)
    parts = [p.strip() for p in parts if p.strip()]
    
    num_re = re.compile(r"^\s*(\d{1,4})\s*$")
    roman_re = re.compile(r"^\s*([IVXLCDMivxlcdm]{1,10})\s*$")
    pageword_re = re.compile(r"^\s*(?:page\s+)?(\d{1,4})\s*$", re.IGNORECASE)
    special_re = re.compile(r"^\s*([A-Z]-\d+)\s*$")
    
    pages = {}
    for i, segment_html in enumerate(parts, start=1):
        text = BeautifulSoup(segment_html, "html.parser").get_text("\n")
        text = re.sub(r"\n{3,}", "\n\n", text).strip("\n")
        lines = text.splitlines()
        
        page_label = None
        last_idx = None

        for idx in range(len(lines) - 1, max(-1, len(lines) - 16), -1):
            s = lines[idx].strip()
            if not s:
                continue
            m = (num_re.match(s) or pageword_re.match(s) or 
                roman_re.match(s) or special_re.match(s))
            if m:
                page_label = m.group(1)
                last_idx = idx
                break
            if len(s) > 100:
                break

        if return_html:
            content = segment_html
        else:
            content = "\n".join(lines[:last_idx]).rstrip() if page_label else text.strip()

        if page_label:
            try:
                key = str(page_to_int(page_label))
            except ValueError:
                print(f"[WARN] Could not parse page label '{page_label}'. Using as-is.")
                key = str(page_label)
        else:
            key = str(i)
        
        if key in pages:
            pages[key] = pages[key] + "\n\n" + content if content else pages[key]
        else:
            pages[key] = content
    
    return pages


def create_pages_dict(
    raw_content: str, 
    file_extension: str, 
    return_html: bool = False
) -> Dict[str, str]:
    """
    Create a dictionary mapping page numbers to content.
    
    Args:
        raw_content: The raw filing content
        file_extension: '.txt' or '.htm'
        return_html (bool): If True and file is .htm, instructs helper
                            to return HTML pages instead of text.
        
    Returns:
        Dictionary mapping page numbers (as strings) to page content
        
    Raises:
        ValueError: If file extension is not supported
    """
    if file_extension.lower() == "txt":
        return pages_by_bottom_number(raw_content)
    elif file_extension.lower() == "htm" or file_extension.lower() == "html":
        return pages_by_bottom_number_html(raw_content, return_html=return_html)
    else:
        raise ValueError(f"Unsupported filing type: {file_extension}")


def extract_section_text(
    raw_content: str,
    toc: Dict[str, str],
    section_name: str,
    file_extension: str,
    return_html: bool = False
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
    try:
        pages_dict = create_pages_dict(
            raw_content, 
            file_extension, 
            return_html=return_html
        )
        start, finish = get_section_page_range(toc, section_name)

        collected_pages = []
        first_page = pages_dict.get(str(start))

        if not first_page:
             print(f"[ERROR] Could not find start page {start} for section '{section_name}'.")
             return ""

        start_index = -1
        if return_html:
            pattern = re.compile(re.escape(section_name), re.IGNORECASE)
            match = pattern.search(first_page)
            if match:
                start_index = match.start()
        else:
            words = map(re.escape, section_name.upper().split())
            pattern = r"\s+".join(words)
            match = re.search(pattern, first_page)
            start_index = match.start() if match is not None else -1

        if start_index == -1:
             print(f"[WARN] Could not find start of section '{section_name}' in page {start}. Using full page.")
             start_index = 0 
        
        first_page_content = first_page[start_index:]

        collected_pages.append(first_page_content)
        for i in range(start + 1, finish):
            page_content = pages_dict.get(str(i))
            if page_content:
                collected_pages.append(page_content)

        sections = list(toc.keys())
        next_header_idx = -1
        try:
            next_header_idx = sections.index(section_name) + 1
        except ValueError:
             print(f"[WARN] Section '{section_name}' not found in TOC keys. Cannot find end of section.")

        if 0 <= next_header_idx < len(sections):
            next_section = sections[next_header_idx]
            final_page = pages_dict.get(str(finish))

            if not final_page:
                 print(f"[WARN] Could not find final page {finish} for section '{section_name}'.")
            else:
                end_index = -1
                if return_html:
                    pattern = re.compile(re.escape(next_section), re.IGNORECASE)
                    match = pattern.search(final_page)
                    if match:
                        end_index = match.start()
                else:
                    words = map(re.escape, next_section.upper().split())
                    pattern = r"\s+".join(words)
                    match = re.search(pattern, final_page)
                    end_index = match.start()

                if end_index == -1:
                    print(f"[WARN] Could not find start of next section '{next_section}' in page {finish}. Using full page.")
                    leak_over_txt = final_page
                else:
                    leak_over_txt = final_page[:end_index]
                    
                collected_pages.append(leak_over_txt)
        else:
            if next_header_idx >= len(sections): 
                 final_page_content = pages_dict.get(str(finish))
                 if final_page_content:
                    collected_pages.append(final_page_content)

        joined_content = "\n\n".join(collected_pages)
        if return_html:
            return joined_content.strip()
        else:
            return re.sub(r'\s+', ' ', joined_content).strip()
    
    except (KeyError, ValueError, Exception) as e:
        print(f"[ERROR] Failed to extract section '{section_name}': {e}")
        return ""
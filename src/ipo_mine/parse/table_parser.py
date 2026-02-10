"""
Table extraction utilities for S1 filings.

This module provides functions to parse HTML filings and extract
tables with proper handling of colspan/rowspan and financial data alignment.
"""

from typing import Dict, List, Tuple
import re
from bs4 import BeautifulSoup


def clean_cell(cell_text: str) -> str:
    """
    Clean and format cell text while preserving formatting.
    """
    text = cell_text.strip().replace("\xa0", " ").replace("\n", " ")
    text = re.sub(r'\s+', ' ', text).strip()
    return text if text else ""


def is_financial_number(text: str) -> bool:
    """
    Check if text is a financial number.
    """
    if not text or not text.strip():
        return False
    
    text = text.strip()
    
    patterns = [
        r'^\$[\d,]+\.?\d*$',           # $10,61,850.00
        r'^\d{1,3}(,\d{3})*$',         # 37,63,392
        r'^\d{1,3}(,\d{3})*\.\d+$',    # 123,456.78
        r'^\(\$?[\d,]+\.?\d*\)$',      # $(123,456) or (123,456)
        r'^-\$?[\d,]+\.?\d*$',         # -$123,456 or -123,456
    ]
    
    return any(re.match(pattern, text) for pattern in patterns)


def build_raw_matrix(table) -> List[List[Dict]]:
    """
    Build raw matrix preserving exact HTML structure.
    """
    matrix = []
    
    for row in table.find_all("tr"):
        row_cells = []
        for cell in row.find_all(["td", "th"]):
            cell_data = {
                'text': clean_cell(cell.get_text()),
                'colspan': int(cell.get("colspan", 1)),
                'rowspan': int(cell.get("rowspan", 1))
            }
            row_cells.append(cell_data)
        matrix.append(row_cells)
    
    return matrix


def expand_matrix_with_spans(raw_matrix: List[List[Dict]]) -> List[List[str]]:
    """
    Convert raw matrix to expanded matrix handling colspan/rowspan.
    """
    if not raw_matrix:
        return []
    
    # Calculate maximum columns needed
    max_cols = 0
    for row in raw_matrix:
        cols = sum(cell['colspan'] for cell in row)
        max_cols = max(max_cols, cols)
    
    # Create expanded matrix
    expanded = []
    rowspan_tracker = {}  # Track cells that span multiple rows
    
    for row_idx, row in enumerate(raw_matrix):
        # Initialize current row
        current_row = [""] * max_cols
        
        # First, fill in any cells from previous rowspans
        for col_idx in range(max_cols):
            if col_idx in rowspan_tracker:
                end_row, text = rowspan_tracker[col_idx]
                if row_idx <= end_row:
                    current_row[col_idx] = text
                if row_idx == end_row:
                    del rowspan_tracker[col_idx]
        
        # Place cells from current row
        col_pos = 0
        for cell in row:
            while col_pos < max_cols and current_row[col_pos] != "":
                col_pos += 1
            
            if col_pos >= max_cols:
                break
            
            # Place the cell
            current_row[col_pos] = cell['text']
            
            # Handle colspan
            for c in range(1, cell['colspan']):
                if col_pos + c < max_cols:
                    current_row[col_pos + c] = ""
            
            # Handle rowspan
            if cell['rowspan'] > 1:
                for c in range(cell['colspan']):
                    if col_pos + c < max_cols:
                        rowspan_tracker[col_pos + c] = (row_idx + cell['rowspan'] - 1, cell['text'])
            
            col_pos += cell['colspan']
        
        expanded.append(current_row)
    
    return expanded


def find_data_columns(matrix: List[List[str]]) -> Dict[str, int]:
    """
    Find exact column positions for financial data headers.
    """
    if not matrix or len(matrix) < 3:
        return {}
    
    data_columns = {}
    
    # Look for headers in first few rows
    for row_idx in range(min(5, len(matrix))):
        row = matrix[row_idx]
        for col_idx, cell in enumerate(row):
            cell_lower = cell.lower().strip()
            
            # Look for specific patterns
            if cell_lower == "actual":
                data_columns['actual'] = col_idx
            elif "pro forma" in cell_lower:
                data_columns['pro_forma'] = col_idx
            elif re.match(r'20\d{2}', cell):  # Year patterns
                data_columns[f'year_{cell}'] = col_idx
    
    return data_columns


def merge_split_dollar_amounts(matrix: List[List[str]]) -> List[List[str]]:
    """
    Merge $ symbols with adjacent numbers.
    """
    if not matrix:
        return matrix
    
    fixed_matrix = [row[:] for row in matrix]
    
    for row_idx, row in enumerate(matrix):
        for col_idx in range(len(row) - 1):
            current_cell = str(row[col_idx]).strip()
            
            # Look for standalone $ followed by a number
            if current_cell == "$":
                # Check next few columns for a number
                for offset in range(1, min(4, len(row) - col_idx)):
                    next_cell = str(row[col_idx + offset]).strip()
                    if next_cell and is_financial_number(next_cell.replace('$', '')):
                        # Merge them
                        if not next_cell.startswith('$'):
                            fixed_matrix[row_idx][col_idx + offset] = f"${next_cell}"
                        fixed_matrix[row_idx][col_idx] = ""
                        break
    
    return fixed_matrix


def realign_table_data(matrix: List[List[str]], data_columns: Dict[str, int]) -> List[List[str]]:
    """
    Realign financial data to proper columns.
    
    Args:
        matrix: Expanded matrix
        data_columns: Dictionary mapping column types to indices
        
    Returns:
        Matrix with realigned data
    """
    if not matrix or not data_columns:
        return matrix
    
    aligned_matrix = []
    
    for row_idx, row in enumerate(matrix):
        # Skip header rows (first 3-4 rows typically)
        if row_idx < 4:
            aligned_matrix.append(row[:])
            continue
        
        # Find row label (first non-empty, non-numeric cell)
        row_label = ""
        label_col = 0
        for col_idx, cell in enumerate(row):
            if cell.strip() and not is_financial_number(cell):
                row_label = cell
                label_col = col_idx
                break
        
        # Collect all financial data from this row
        financial_data = []
        for cell in row:
            if is_financial_number(cell):
                financial_data.append(cell)
        
        # Create new row with proper alignment
        new_row = [""] * len(row)
        new_row[label_col] = row_label 
        
        # Place financial data in correct columns
        if financial_data:
            # Get sorted data column positions
            sorted_data_cols = []
            if 'actual' in data_columns:
                sorted_data_cols.append(('actual', data_columns['actual']))
            if 'pro_forma' in data_columns:
                sorted_data_cols.append(('pro_forma', data_columns['pro_forma']))
            
            # Add any year columns
            for key, col in data_columns.items():
                if key.startswith('year_'):
                    sorted_data_cols.append((key, col))
            
            # Sort by column index
            sorted_data_cols.sort(key=lambda x: x[1])
            for i, (col_type, col_idx) in enumerate(sorted_data_cols):
                if i < len(financial_data) and col_idx < len(new_row):
                    new_row[col_idx] = financial_data[i]
        
        aligned_matrix.append(new_row)
    
    return aligned_matrix


def process_table(table, align_data: bool = True) -> List[List[str]]:
    """
    Process a single table through the full pipeline.
    
    Args:
        table: BeautifulSoup table element
        align_data: Whether to perform data alignment for financial tables
        
    Returns:
        Processed matrix as list of lists
    """
    # Build raw matrix
    raw_matrix = build_raw_matrix(table)
    if not raw_matrix:
        return []
    
    # Expand matrix handling spans
    matrix = expand_matrix_with_spans(raw_matrix)
    if not matrix:
        return []
    
    # Merge split dollar amounts
    matrix = merge_split_dollar_amounts(matrix)
    
    # Optionally realign data to proper columns
    if align_data:
        data_columns = find_data_columns(matrix)
        if data_columns:
            matrix = realign_table_data(matrix, data_columns)
    
    # Clean up empty rows
    matrix = [row for row in matrix if any(cell.strip() for cell in row)]
    
    return matrix


def extract_tables_from_html(html_content: str, align_data: bool = True) -> List[List[List[str]]]:
    """
    Extract all tables from HTML content.
    
    Args:
        html_content: Raw HTML content
        align_data: Whether to perform data alignment for financial tables
        
    Returns:
        List of processed tables, each as a matrix (list of lists)
    """
    soup = BeautifulSoup(html_content, "html.parser")
    tables = soup.find_all("table")
    
    extracted_tables = []
    for table in tables:
        matrix = process_table(table, align_data=align_data)
        if matrix:
            extracted_tables.append(matrix)
    
    return extracted_tables


def extract_table_by_index(
    html_content: str,
    table_index: int,
    align_data: bool = True
) -> List[List[str]]:
    """
    Extract a specific table by index.
    
    Args:
        html_content: Raw HTML content
        table_index: 0-based index of table to extract
        align_data: Whether to perform data alignment for financial tables
        
    Returns:
        Processed table matrix
        
    Raises:
        IndexError: If table index is out of range
    """
    soup = BeautifulSoup(html_content, "html.parser")
    tables = soup.find_all("table")
    
    if table_index < 0 or table_index >= len(tables):
        raise IndexError(
            f"Table index {table_index} out of range. "
            f"Found {len(tables)} tables (0-{len(tables)-1})."
        )
    
    return process_table(tables[table_index], align_data=align_data)
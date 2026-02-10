from enum import Enum

class FormType(Enum):
    """SEC form types for IPO filings."""
    S1_A = "S-1/A"
    S1 = "S-1"
    F1_A = "F-1/A"
    F1 = "F-1"
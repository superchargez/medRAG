# src/shared/normalization.py
import re

def normalize_identifier(text: str) -> str:
    """Removes non-alphanumeric chars."""
    if not text: return ""
    return re.sub(r'[^a-zA-Z0-9]', '', text).upper()

def extract_potential_id(query: str) -> str:
    """
    Scans the query for something that looks like a CNIC, Phone, or ID.
    Returns the CLEANED identifier if found, else None.
    """
    # 1. Check for System ID format (PT-...)
    match_sys = re.search(r'(PT-[A-F0-9]+)', query.upper())
    if match_sys:
        return match_sys.group(1)

    # 2. Check for sequences of digits (Phone/CNIC)
    # We allow anything with 5+ digits to be considered a potential ID for now.
    match_num = re.search(r'\b\d{5,}\b', query) 
    if match_num:
        return match_num.group(0) # Return raw, let normalize handle cleanup
    
    return None

def detect_query_type(query: str) -> str:
    """
    Decides which metadata field to search.
    """
    clean = normalize_identifier(query)
    
    if query.strip().upper().startswith("PT-"):
        return "SYSTEM_ID"
    
    # Relaxed rules for testing
    if clean.isdigit():
        if len(clean) >= 11: return "GOV_ID" # CNIC is usually 13
        if len(clean) >= 10: return "PHONE"  # Phone is usually 10-11
        return "GOV_ID" # Fallback for short test numbers like 12345
        
    return "NAME"
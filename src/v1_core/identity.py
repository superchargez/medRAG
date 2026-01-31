# src\core\identity.py

from langchain_chroma import Chroma
from typing import List, Dict, Any
from src.shared.normalization import normalize_identifier
import re

def resolve_patient_identity(user_query: str, patient_store: Chroma, threshold: float = 0.7) -> Dict[str, Any]:
    """
    Robust Identity Resolution with "Greedy" Search.
    
    Logic:
    1. Clean the input.
    2. If it looks like a System ID -> Check ID field.
    3. If it contains numbers -> Check BOTH GovID AND Phone fields.
    4. If it contains text -> Check Name field (Raw & Title Case) -> Then Fuzzy Search.
    """
    if not user_query:
         return {"status": "NOT_FOUND", "matches": []}

    # 1. Prepare Data
    clean_query = normalize_identifier(user_query)
    raw_query = user_query.strip()
    
    print(f"   [Identity] Resolving: '{raw_query}' (Clean: {clean_query})")
    
    # Store unique candidates here (Key = Patient ID)
    # Format: { "PT-123": {id, name, score} }
    candidates = {}

    # --- STRATEGY A: SYSTEM ID (PT-XXX) ---
    if "PT-" in raw_query.upper():
        print(f"   [Identity] Checking System ID...")
        # Extract just the ID part if user typed extra text
        match = re.search(r'(PT-[A-F0-9]+)', raw_query.upper())
        if match:
            sys_id = match.group(1)
            results = patient_store.get(where={"patient_id": sys_id})
            _add_metadata_matches(results, candidates, score=0.0)

    # --- STRATEGY B: NUMERIC SEARCH (Gov ID OR Phone) ---
    # If the clean query has digits, we check ALL numeric fields.
    # We don't care if it "looks like" a phone or CNIC. We check both.
    if clean_query.isdigit():
        print(f"   [Identity] Checking Numeric Fields (GovID & Phone)...")
        
        # 1. Check Gov ID
        results_gov = patient_store.get(where={"gov_id": clean_query})
        _add_metadata_matches(results_gov, candidates, score=0.0)
        
        # 2. Check Phone (Contact Number)
        results_phone = patient_store.get(where={"contact_number": clean_query})
        _add_metadata_matches(results_phone, candidates, score=0.0)

    # --- STRATEGY C: EXACT NAME SEARCH (Case Handling) ---
    # Only if we haven't found a System ID match yet (names are less precise)
    if not candidates and any(c.isalpha() for c in raw_query):
        print(f"   [Identity] Checking Metadata Name...")
        
        # 1. Try Exact Match (e.g. "Ali Khan")
        results_name = patient_store.get(where={"patient_name": raw_query})
        _add_metadata_matches(results_name, candidates, score=0.1)
        
        # 2. Try Title Case (e.g. "sara khan" -> "Sara Khan")
        # This fixes the issue you saw where lowercase failed.
        if raw_query.title() != raw_query:
            results_title = patient_store.get(where={"patient_name": raw_query.title()})
            _add_metadata_matches(results_title, candidates, score=0.1)

    # --- DECISION POINT ---
    # If we found exact metadata matches (ID, Number, or Name), we might be done.
    # But if we found NOTHING, we fall back to Fuzzy.
    
    if not candidates:
        print(f"   [Identity] No metadata match. Falling back to fuzzy search...")
        # Note: We use the RAW query for vectors (embeddings handle casing better)
        results_vec = patient_store.similarity_search_with_score(raw_query, k=10)
        
        for doc, score in results_vec:
            if score > threshold: continue
            
            pid = doc.metadata.get("patient_id")
            pname = doc.metadata.get("patient_name", "Unknown")
            
            if pid and pid not in candidates:
                candidates[pid] = {"id": pid, "name": pname, "score": score}
            elif pid and score < candidates[pid]["score"]:
                candidates[pid]["score"] = score # Update with better score

    # --- FINAL PROCESSING ---
    sorted_matches = sorted(candidates.values(), key=lambda x: x['score'])
    
    if not sorted_matches:
        return {"status": "NOT_FOUND", "matches": []}
    
    if len(sorted_matches) == 1:
        best = sorted_matches[0]
        print(f"   [Identity] Unique Match: {best['name']} ({best['id']})")
        return {"status": "FOUND", "id": best['id'], "name": best['name']}
    
    print(f"   [Identity] Ambiguous. Found {len(sorted_matches)} candidates.")
    return {"status": "AMBIGUOUS", "matches": sorted_matches}

def _add_metadata_matches(chroma_results, candidates_dict, score):
    """Helper to parse Chroma .get() results and add to dictionary."""
    if chroma_results and len(chroma_results['ids']) > 0:
        for meta in chroma_results['metadatas']:
            pid = meta.get("patient_id")
            name = meta.get("patient_name")
            if pid:
                # We give metadata matches a very low score (0.0 or 0.1) so they rank top
                candidates_dict[pid] = {"id": pid, "name": name, "score": score}
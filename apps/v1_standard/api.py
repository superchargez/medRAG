# api.py
from fastapi import FastAPI
import chromadb
from langchain_chroma import Chroma
import uvicorn

from pydantic import BaseModel
from langchain_core.documents import Document
from src.shared.config import INDEXING_BATCH_SIZE

# --- IMPORTS ---
from src.shared.config import *
from src.shared.embedding import RemoteEmbeddingFunction
from src.shared.llm import get_cerebras_client
from src.v1_core.router import get_routing_decision
from src.v1_core.identity import resolve_patient_identity
from src.v1_core.patients_ import get_patient_context
from src.shared.wiki_ import get_wiki_context
from src.prompts.templates import SAFETY_SYNTHESIS_PROMPT
from src.shared.normalization import normalize_identifier
from src.shared.normalization import extract_potential_id

app = FastAPI(title="MediFlow RAG API")

# --- INITIALIZATION ---
embed_fn = RemoteEmbeddingFunction(api_url=EMBEDDING_API_URL)
chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

wiki_store = Chroma(
    client=chroma_client, 
    collection_name=COLLECTION_WIKI, 
    embedding_function=embed_fn
)
patient_store = Chroma(
    client=chroma_client, 
    collection_name=COLLECTION_PATIENTS, 
    embedding_function=embed_fn
)

cerebras_client = get_cerebras_client()

import re
@app.get("/ask")
async def ask_question(query: str):
    print(f"\n[QUERY] {query}")
    
    # 1. Route
    decision = get_routing_decision(query)
    
    # --- LOGIC FIX: OVERRIDE GENERAL IF ID DETECTED ---
    # Sometimes LLM thinks "who is 12345" is general math. We correct it.
    potential_id = extract_potential_id(query)
    if decision.route == "general" and potential_id:
        print(f"   [Router Override] Found ID '{potential_id}' in General query. Switching to patient_info.")
        decision.route = "patient_info"
        decision.patient_name = potential_id

    print(f"[DECISION] {decision.route} | Target: {decision.patient_name}")

    # 2. General Query
    if decision.route == "general":
        # ... existing general logic ...
        pass

    # 3. Patient Identity Resolution
    p_id = None
    patient_context = ""
    
    if decision.route in ["patient_info", "clinical_safety"]:
        
        target = decision.patient_name
        
        # Fallback: If LLM failed to extract a target but routed to patient_info, 
        # use the potential_id or the whole query (if short)
        if not target:
            if potential_id:
                target = potential_id
            elif len(query.split()) <= 3: # E.g. "Ali Khan"
                target = query
        
        if not target:
             return {"answer": "I understood you are asking about a patient, but I couldn't identify who. Please specify a Name, ID, CNIC, or Phone."}

        # --- THE UNIVERSAL LOOKUP ---
        # We pass 'target' (which could be "Ali", "12345", "PT-XXX") to identity.py
        # identity.py already has the Cascade logic to handle all these types.
        identity_result = resolve_patient_identity(target, patient_store)
        
        if identity_result["status"] == "NOT_FOUND":
            return {"error": f"Patient identifier '{target}' not found in database."}
        
        if identity_result["status"] == "AMBIGUOUS":
            return {
                "action": "CLARIFICATION_NEEDED", 
                "message": f"Multiple records found for '{target}'.",
                "candidates": identity_result["matches"]
            }
        
        # FOUND
        p_id = identity_result["id"]
        found_name = identity_result["name"]
        print(f"   [Identity] Resolved '{target}' -> {found_name} ({p_id})")
        
        # Fetch Context
        patient_context = get_patient_context(p_id, query, patient_store) 

    # 4. Medical Knowledge Retrieval
    wiki_context = ""
    if decision.route in ["medical_wiki", "clinical_safety"]:
        wiki_context = get_wiki_context(decision.medical_terms, wiki_store)

    # 5. Synthesis (The Safety Check)
    final_prompt = SAFETY_SYNTHESIS_PROMPT.format(
        patient_data=patient_context,
        wiki_data=wiki_context,
        query=query
    )

    resp = cerebras_client.chat.completions.create(
        messages=[{"role": "user", "content": final_prompt}],
        model=LLM_MODEL_ID,
        temperature=0.2
    )

    return {
        "answer": resp.choices[0].message.content,
        "route": decision.route,
        "patient_id": p_id,
        "sources_used": {
            "patient_db": bool(patient_context),
            "wiki_db": bool(wiki_context)
        }
    }

# 1. Update Schema
class PatientRecord(BaseModel):
    # Identifiers
    patient_id: str = ""        # System ID (PT-XXX)
    patient_name: str
    gov_id: str = ""            # CNIC / Passport (Unique)
    contact_number: str = ""    # Family Phone (Shared)
    
    # Medical Data
    date: str
    condition: str
    note_content: str
    record_type: str = "General"

@app.post("/ingest_record")
async def ingest_record(record: PatientRecord):
    print(f"\n[INGEST] Processing: {record.patient_name}")
    
    # A. NORMALIZE DATA
    clean_gov_id = normalize_identifier(record.gov_id)
    clean_phone = normalize_identifier(record.contact_number)
    
    # B. LOGIC 1: VALIDATE GOV ID (CNIC) - Must be Unique
    if clean_gov_id:
        # Check if this CNIC belongs to someone else
        existing_cnic = patient_store.get(where={"gov_id": clean_gov_id})
        
        if existing_cnic and len(existing_cnic['ids']) > 0:
            existing_meta = existing_cnic['metadatas'][0]
            stored_name = existing_meta.get("patient_name", "Unknown")
            stored_pid = existing_meta.get("patient_id")
            
            # If names don't match, block it.
            # (Allows updating the SAME person, blocks creating a duplicate person)
            if stored_name.strip().lower() != record.patient_name.strip().lower():
                msg = f"⛔ CONFLICT: CNIC {record.gov_id} is already registered to '{stored_name}' ({stored_pid})."
                print(f"   > {msg}")
                return {"status": "error", "message": msg}
            
            # If name matches, we assume it's the same person, so we use their existing PID
            if not record.patient_id:
                record.patient_id = stored_pid
                print(f"   > identified existing patient via CNIC: {stored_pid}")

    # C. LOGIC 2: AUTO-GENERATE SYSTEM ID
    if not record.patient_id:
        # Generate new ID
        import uuid
        new_id = f"PT-{uuid.uuid4().hex[:6].upper()}"
        record.patient_id = new_id
        print(f"   > Generated New System ID: {new_id}")

    # D. PREPARE DOCUMENT
    # We add the identifiers to the text so they are searchable via vector too (fallback)
    full_text = (
        f"Patient: {record.patient_name}. "
        f"ID: {record.patient_id}. "
        f"CNIC: {clean_gov_id}. "
        f"Phone: {clean_phone}. "
        f"Date: {record.date}. "
        f"Condition: {record.condition}. "
        f"Note: {record.note_content}"
    )
    
    doc = Document(
        page_content=full_text,
        metadata={
            "source": f"{record.patient_id}_{record.date}_{hash(full_text)}",
            "patient_id": record.patient_id,
            "patient_name": record.patient_name,
            "gov_id": clean_gov_id,       # Saved Clean
            "phone": clean_phone,         # Saved Clean
            "date": record.date,
            "condition": record.condition,
            "type": record.record_type
        }
    )

    try:
        patient_store.add_documents([doc])
        return {
            "status": "success", 
            "patient_id": record.patient_id,
            "message": f"Saved record for {record.patient_name}"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
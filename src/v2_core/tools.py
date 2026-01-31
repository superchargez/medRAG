# src/v2_core/tools.py
import json
from datetime import datetime
from typing import Optional, Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
import chromadb
from langchain_chroma import Chroma

# Imports from Shared Config
from src.shared.config import (
    CHROMA_HOST, CHROMA_PORT, 
    COLLECTION_PATIENTS, COLLECTION_WIKI, 
    EMBEDDING_API_URL
)
from src.shared.embedding import RemoteEmbeddingFunction
from src.v1_core.identity import resolve_patient_identity
from src.v1_core.patients_ import get_patient_context
from src.shared.wiki_ import get_wiki_context

# --- INITIALIZATION ---
embed_fn = RemoteEmbeddingFunction(api_url=EMBEDDING_API_URL)
chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

patient_store = Chroma(
    client=chroma_client, 
    collection_name=COLLECTION_PATIENTS, 
    embedding_function=embed_fn
)
wiki_store = Chroma(
    client=chroma_client, 
    collection_name=COLLECTION_WIKI, 
    embedding_function=embed_fn
)

# --- 1. IDENTITY TOOL ---
class IdentityInput(BaseModel):
    query: str = Field(description="Name, ID, Phone, or CNIC.")

class IdentityTool(BaseTool):
    name: str = "lookup_patient_id"
    description: str = "Find a patient's System ID (PT-XXX). Use this FIRST if you only have a name."
    args_schema: Type[BaseModel] = IdentityInput

    def _run(self, query: str):
        result = resolve_patient_identity(query, patient_store)
        if result["status"] == "FOUND":
            return json.dumps({"status": "found", "id": result["id"], "name": result["name"]})
        elif result["status"] == "AMBIGUOUS":
            return json.dumps({"status": "ambiguous", "candidates": [m["name"] for m in result["matches"]]})
        return json.dumps({"status": "not_found"})

# --- 2. HISTORY TOOL ---
class HistoryInput(BaseModel):
    patient_id: str = Field(description="System ID (PT-XXX).")

class PatientHistoryTool(BaseTool):
    name: str = "get_patient_medical_history"
    description: str = "Get text records/notes for a specific ID."
    args_schema: Type[BaseModel] = HistoryInput

    def _run(self, patient_id: str):
        return get_patient_context(patient_id, "history", patient_store)

# --- 3. ANALYTICS / POPULATION TOOL (The Brain) ---
class AnalyticsInput(BaseModel):
    condition: Optional[str] = Field(None, description="Disease e.g. 'Asthma'")
    gender: Optional[str] = Field(None, description="'Male' or 'Female'")
    min_age: Optional[int] = Field(None, description="Min Age")
    max_age: Optional[int] = Field(None, description="Max Age")
    days_ago: Optional[int] = Field(None, description="Look back X days")

class PatientAnalyticsTool(BaseTool):
    name: str = "patient_analytics_tool"
    description: str = "Returns JSON list of patients matching criteria. Use for counting, listing, or stats."
    args_schema: Type[BaseModel] = AnalyticsInput

    def _run(self, condition: str=None, gender: str=None, min_age: int=None, max_age: int=None, days_ago: int=None):
        # Build Filter
        where_clauses = []
        
        if condition: where_clauses.append({"condition": condition})
        if gender: where_clauses.append({"gender": gender})
        
        # Date Filter (String Comparison works for ISO dates)
        if days_ago:
            from datetime import timedelta
            cutoff = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
            where_clauses.append({"date": {"$gte": cutoff}})

        final_where = {}
        if len(where_clauses) == 1: final_where = where_clauses[0]
        elif len(where_clauses) > 1: final_where = {"$and": where_clauses}

        # Fetch Metadata
        try:
            results = patient_store.get(where=final_where, limit=100)
        except Exception as e:
            return f"DB Error: {e}"

        if not results['ids']:
            return json.dumps({"count": 0, "records": []})

        # Python-side Filtering (Age) & Formatting
        records = []
        for meta in results['metadatas']:
            p_age = meta.get("age", 0)
            if min_age and p_age < min_age: continue
            if max_age and p_age > max_age: continue
            
            records.append({
                "name": meta.get("patient_name"),
                "id": meta.get("patient_id"),
                "age": p_age,
                "gender": meta.get("gender"),
                "condition": meta.get("condition"),
                "date": meta.get("date")
            })

        return json.dumps({
            "count": len(records),
            "records": records
        }, indent=2)

# --- 4. WIKI TOOL ---
class WikiInput(BaseModel):
    query: str = Field(description="Medical term.")

class MedicalWikiTool(BaseTool):
    name: str = "check_drug_info"
    description: str = "Search medical knowledge base."
    args_schema: Type[BaseModel] = WikiInput

    def _run(self, query: str):
        return get_wiki_context(query.split(), wiki_store)
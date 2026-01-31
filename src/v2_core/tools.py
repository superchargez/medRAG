# src/v2_core/tools.py
import json
from datetime import datetime
from typing import Optional, Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from sqlalchemy import select, and_
from sqlalchemy.orm import Session

# Imports
from src.v2_core.database import SessionLocal, Patient, ClinicalNote, MedicalKnowledge
from src.shared.config import EMBEDDING_API_URL
from src.shared.embedding import RemoteEmbeddingFunction

embed_fn = RemoteEmbeddingFunction(api_url=EMBEDDING_API_URL)

def get_db_session():
    return SessionLocal()

# --- 1. IDENTITY TOOL (Best for "Who is X?" or "Get ID for X") ---
class IdentityInput(BaseModel):
    query: str = Field(description="Name, ID, Phone, or CNIC.")

class IdentityTool(BaseTool):
    name: str = "lookup_patient_id"
    description: str = "Find a specific patient's System ID (PT-XXX). Use this for 'Who is...' or 'Find patient...'."
    args_schema: Type[BaseModel] = IdentityInput

    def _run(self, query: str):
        db = get_db_session()
        try:
            # Exact ID
            p = db.scalar(select(Patient).where(Patient.system_id == query.upper()))
            if p: return json.dumps({"status": "found", "id": p.system_id, "name": p.full_name})

            # Fuzzy Name
            results = db.scalars(select(Patient).where(Patient.full_name.ilike(f"%{query}%"))).all()
            
            if not results: return json.dumps({"status": "not_found"})
            if len(results) == 1: return json.dumps({"status": "found", "id": results[0].system_id, "name": results[0].full_name})
            
            return json.dumps({"status": "ambiguous", "candidates": [{"name": r.full_name, "id": r.system_id} for r in results]})
        finally:
            db.close()

# --- 2. ANALYTICS TOOL (Best for "How many?", "List...", "Stats") ---
class AnalyticsInput(BaseModel):
    name_filter: Optional[str] = Field(None, description="Filter by partial name (e.g. 'Ali')")
    condition: Optional[str] = Field(None, description="Filter by disease (e.g. 'Asthma')")
    gender: Optional[str] = Field(None, description="'Male' or 'Female'")
    start_date: Optional[str] = Field(None, description="YYYY-MM-DD")
    end_date: Optional[str] = Field(None, description="YYYY-MM-DD")

class PatientAnalyticsTool(BaseTool):
    name: str = "patient_analytics_tool"
    description: str = "Run SQL stats. Use for 'How many...', 'List patients...', 'Who visited...'. Can filter by Name, Condition, Gender, Date."
    args_schema: Type[BaseModel] = AnalyticsInput

    def _run(self, name_filter: str=None, condition: str=None, gender: str=None, start_date: str=None, end_date: str=None):
        db = get_db_session()
        try:
            # Start with a Join
            stmt = select(Patient.full_name, Patient.system_id, Patient.gender, ClinicalNote.condition, ClinicalNote.visit_date)\
                .join(ClinicalNote, Patient.id == ClinicalNote.patient_id)
            
            filters = []
            
            # --- NEW: Name Filter ---
            if name_filter:
                filters.append(Patient.full_name.ilike(f"%{name_filter}%"))
            
            if gender: filters.append(Patient.gender == gender)
            if condition: filters.append(ClinicalNote.condition == condition)
            if start_date: filters.append(ClinicalNote.visit_date >= datetime.strptime(start_date, "%Y-%m-%d").date())
            if end_date: filters.append(ClinicalNote.visit_date <= datetime.strptime(end_date, "%Y-%m-%d").date())
            
            if filters: stmt = stmt.where(and_(*filters))
            
            # Execute
            results = db.execute(stmt).all()
            
            # Deduplicate patients if counting people (since one patient might have multiple notes)
            unique_patients = {}
            for r in results:
                if r.system_id not in unique_patients:
                    unique_patients[r.system_id] = {
                        "name": r.full_name,
                        "id": r.system_id,
                        "gender": r.gender,
                        "latest_condition": r.condition,
                        "latest_date": str(r.visit_date)
                    }
            
            records = list(unique_patients.values())
            
            return json.dumps({
                "count": len(records),
                "summary": f"Found {len(records)} unique patients.",
                "records": records[:20]
            }, indent=2)
            
        except Exception as e:
            return f"SQL Error: {str(e)}"
        finally:
            db.close()

# --- 3. HISTORY TOOL ---
class HistoryInput(BaseModel):
    patient_id: str = Field(description="System ID (PT-XXX)")
    query: Optional[str] = Field(None, description="Specific symptom to search for (e.g. 'chest pain')")

class PatientHistoryTool(BaseTool):
    name: str = "get_patient_medical_history"
    description: str = "Get records. If 'query' is provided, performs Vector Search on notes."
    args_schema: Type[BaseModel] = HistoryInput

    def _run(self, patient_id: str, query: str = None):
        db = get_db_session()
        try:
            p = db.scalar(select(Patient).where(Patient.system_id == patient_id))
            if not p: return "Patient ID not found."

            stmt = select(ClinicalNote).where(ClinicalNote.patient_id == p.id)

            if query:
                query_vec = embed_fn.embed_query(query)
                stmt = stmt.order_by(ClinicalNote.embedding.cosine_distance(query_vec))
            else:
                stmt = stmt.order_by(ClinicalNote.visit_date.desc())

            notes = db.scalars(stmt.limit(5)).all()
            return "\n".join([f"Date: {n.visit_date} | Cond: {n.condition} | Note: {n.note_content}" for n in notes]) if notes else "No records."
        finally:
            db.close()

# --- 4. WIKI TOOL ---
class WikiInput(BaseModel):
    query: str = Field(description="Medical term or question.")

class MedicalWikiTool(BaseTool):
    name: str = "check_drug_info"
    description: str = "Search medical knowledge base (Wiki) using Vector Search."
    args_schema: Type[BaseModel] = WikiInput

    def _run(self, query: str):
        db = get_db_session()
        try:
            query_vec = embed_fn.embed_query(query)
            stmt = select(MedicalKnowledge).order_by(
                MedicalKnowledge.embedding.cosine_distance(query_vec)
            ).limit(3)
            results = db.scalars(stmt).all()
            if not results: return "No relevant medical info found."
            return "\n\n".join([f"[Topic: {r.topic}] {r.content}" for r in results])
        finally:
            db.close()
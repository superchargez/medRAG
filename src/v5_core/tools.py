# src/v5_core/tools.py
import json
from datetime import datetime, timedelta
from typing import Type, List, Optional
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from src.shared.config import EMBEDDING_API_URL
from src.shared.embedding import RemoteEmbeddingFunction
from src.v5_core.database import get_connection, close_connection

embed_fn = RemoteEmbeddingFunction(api_url=EMBEDDING_API_URL)

# --- TOOL 1: COHORT FINDER (Structured Filters) ---
class CohortInput(BaseModel):
    gender: Optional[str] = Field(None, description="Male or Female")
    min_age: Optional[int] = Field(None, description="Minimum age in years")
    max_age: Optional[int] = Field(None, description="Maximum age in years")
    visited_within_months: Optional[int] = Field(None, description="Filter patients who visited in the last X months")

class CohortFinderTool(BaseTool):
    name: str = "find_patient_cohort"
    description: str = (
        "Finds a list of patients matching demographic and visit criteria. "
        "Use this for questions like 'Male patients over 40 who visited recently'. "
        "Returns a list of Patient IDs and Names."
    )
    args_schema: Type[BaseModel] = CohortInput

    def _run(self, gender: str = None, min_age: int = None, max_age: int = None, visited_within_months: int = None):
        conn = get_connection()
        cur = conn.cursor()
        try:
            query = """
                SELECT DISTINCT p.id, p.full_name, p.dob, p.gender 
                FROM patients p
                JOIN clinical_notes cn ON p.id = cn.patient_id
                WHERE 1=1
            """
            params = []

            # 1. Gender Filter
            if gender:
                query += " AND p.gender = ?"
                params.append(gender)

            # 2. Age Filter (Calculate DOB thresholds)
            today = datetime.now()
            if min_age:
                # Born before today - min_age
                max_dob = (today - timedelta(days=min_age*365)).strftime("%Y-%m-%d")
                query += " AND p.dob <= ?"
                params.append(max_dob)
            if max_age:
                # Born after today - max_age
                min_dob = (today - timedelta(days=max_age*365)).strftime("%Y-%m-%d")
                query += " AND p.dob >= ?"
                params.append(min_dob)

            # 3. Visit Date Filter
            if visited_within_months:
                # Visited after today - X months
                cutoff_date = (today - timedelta(days=visited_within_months*30)).strftime("%Y-%m-%d")
                query += " AND cn.visit_date >= ?"
                params.append(cutoff_date)

            print(f"   [Cohort Tool] SQL: {query} | Params: {params}")
            cur.execute(query, params)
            rows = cur.fetchall()

            if not rows:
                return "No patients found matching these criteria."

            # Return structured list
            results = []
            for r in rows:
                results.append({"id": r[0], "name": r[1], "dob": r[2], "gender": r[3]})
            
            return json.dumps(results)

        except Exception as e:
            return f"Error finding cohort: {e}"
        finally:
            cur.close()
            close_connection(conn)

# --- TOOL 2: CONDITION ANALYZER (Vector Search on Specific Patients) ---
class ConditionInput(BaseModel):
    patient_ids: List[int] = Field(description="List of patient IDs to check (from cohort tool).")
    medical_concept: str = Field(description="The condition/symptom to search for (e.g., 'Asthma', 'Heart Failure').")

class PatientConditionTool(BaseTool):
    name: str = "analyze_patient_condition"
    description: str = (
        "Checks specific patients' notes for a medical condition using Vector Search. "
        "Use this to see which patients in a cohort have a specific disease or symptom. "
        "Returns the relevant note snippet and a match score for each patient."
    )
    args_schema: Type[BaseModel] = ConditionInput

    def _run(self, patient_ids: List[int], medical_concept: str):
        if not patient_ids:
            return "No patient IDs provided."

        conn = get_connection()
        cur = conn.cursor()
        try:
            # Embed the concept
            query_vector = embed_fn.embed_query(medical_concept)
            query_vec_str = json.dumps(query_vector)

            # Dynamic SQL to filter by the specific IDs provided
            placeholders = ','.join(['?'] * len(patient_ids))
            
            # We select the note AND the distance
            sql = f"""
                SELECT p.id, p.full_name, cn.note_content, 
                       vector_distance_cos(cn.embedding, vector32(?)) as distance
                FROM clinical_notes cn
                JOIN patients p ON p.id = cn.patient_id
                WHERE p.id IN ({placeholders})
                ORDER BY distance ASC
            """
            
            # Params: vector first, then the list of IDs
            params = [query_vec_str] + patient_ids
            
            cur.execute(sql, params)
            rows = cur.fetchall()

            results = []
            for r in rows:
                p_id, name, content, dist = r
                # Threshold: < 0.35 is usually a good semantic match
                match_status = "HIGH MATCH" if dist < 0.35 else "LOW/NO MATCH"
                
                # Only return if it's somewhat relevant (e.g., < 0.5) to save tokens
                if dist < 0.5:
                    results.append({
                        "patient": f"{name} (ID: {p_id})",
                        "status": match_status,
                        "relevance_score": round(1 - dist, 2), # Convert distance to similarity
                        "note_excerpt": content[:200] + "..." # Truncate for brevity
                    })

            if not results:
                return f"None of the provided patients have notes matching '{medical_concept}'."

            return json.dumps(results, indent=2)

        except Exception as e:
            return f"Error analyzing conditions: {e}"
        finally:
            cur.close()
            close_connection(conn)

# --- TOOL 3: DRUG SAFETY (Wiki) ---
class DrugSafetyInput(BaseModel):
    drug_name: str = Field(description="Name of the drug.")

class DrugSafetyTool(BaseTool):
    name: str = "check_drug_safety"
    description: str = "Search medical knowledge for drug contraindications or safety info."
    args_schema: Type[BaseModel] = DrugSafetyInput

    def _run(self, drug_name: str):
        query = f"{drug_name} contraindications safety"
        query_vector = embed_fn.embed_query(query)
        query_vec_str = json.dumps(query_vector)
        
        conn = get_connection()
        cur = conn.cursor()
        try:
            cur.execute("""
                SELECT topic, content, vector_distance_cos(embedding, vector32(?)) as distance
                FROM medical_knowledge
                ORDER BY distance ASC
                LIMIT 2
            """, (query_vec_str,))
            
            rows = cur.fetchall()
            if not rows:
                return f"No information found for {drug_name}."
            
            return f"--- Safety Info for {drug_name} ---\n" + "\n".join([f"- {r[1]}" for r in rows])
            
        finally:
            cur.close()
            close_connection(conn)

# --- TOOL 4: IDENTITY LOOKUP ---
class IdentityInput(BaseModel):
    name_or_id: str = Field(description="Patient Name or System ID")

class IdentityTool(BaseTool):
    name: str = "lookup_patient_identity"
    description: str = "Find a specific patient's ID by name. Use this if the user asks about a specific person."
    args_schema: Type[BaseModel] = IdentityInput

    def _run(self, name_or_id: str):
        conn = get_connection()
        cur = conn.cursor()
        try:
            cur.execute("SELECT id, full_name, system_id, dob FROM patients WHERE full_name LIKE ? OR system_id = ?", 
                        (f"%{name_or_id}%", name_or_id))
            rows = cur.fetchall()
            if not rows: return "Patient not found."
            return json.dumps([{"id": r[0], "name": r[1], "sys_id": r[2], "dob": r[3]} for r in rows])
        finally:
            cur.close()
            close_connection(conn)
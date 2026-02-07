# src/v5_core/tools.py
import json
import sys
from datetime import datetime, timedelta
from typing import Type, List, Optional
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from src.shared.config import EMBEDDING_API_URL
from src.shared.embedding import RemoteEmbeddingFunction
from src.v5_core.database import get_connection, close_connection

embed_fn = RemoteEmbeddingFunction(api_url=EMBEDDING_API_URL)

# --- HELPER FOR LOGGING ---
def log_action(action_type: str, content: str):
    """Prints color-coded logs to the terminal for visibility."""
    # Cyan for SQL, Green for Vector, Yellow for Tool Logic
    colors = {
        "SQL": "\033[96m",    # Cyan
        "VECTOR": "\033[92m", # Green
        "TOOL": "\033[93m",   # Yellow
        "RESET": "\033[0m"
    }
    print(f"{colors.get(action_type, '')}[{action_type}] {content}{colors['RESET']}")
    sys.stdout.flush() # Force print immediately

# --- TOOL 1: COHORT FINDER ---
class CohortInput(BaseModel):
    gender: Optional[str] = Field(None, description="Male or Female")
    min_age: Optional[int] = Field(None, description="Minimum age")
    max_age: Optional[int] = Field(None, description="Maximum age")
    visited_within_months: Optional[int] = Field(None, description="Visited in last X months")

class CohortFinderTool(BaseTool):
    name: str = "find_patient_cohort"
    description: str = "Finds patients matching demographic/visit criteria. Returns IDs."
    args_schema: Type[BaseModel] = CohortInput

    def _run(self, gender: str = None, min_age: int = None, max_age: int = None, visited_within_months: int = None):
        log_action("TOOL", f"Finding cohort: Gender={gender}, Age={min_age}-{max_age}, Visit={visited_within_months}mo")
        
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

            if gender:
                query += " AND p.gender = ?"
                params.append(gender)

            today = datetime.now()
            if min_age:
                max_dob = (today - timedelta(days=min_age*365)).strftime("%Y-%m-%d")
                query += " AND p.dob <= ?"
                params.append(max_dob)
            if max_age:
                min_dob = (today - timedelta(days=max_age*365)).strftime("%Y-%m-%d")
                query += " AND p.dob >= ?"
                params.append(min_dob)
            if visited_within_months:
                cutoff_date = (today - timedelta(days=visited_within_months*30)).strftime("%Y-%m-%d")
                query += " AND cn.visit_date >= ?"
                params.append(cutoff_date)

            log_action("SQL", f"Executing: {query} | Params: {params}")
            
            cur.execute(query, params)
            rows = cur.fetchall()

            if not rows:
                log_action("TOOL", "No results found.")
                return "No patients found matching these criteria."

            results = [{"id": r[0], "name": r[1], "dob": r[2], "gender": r[3]} for r in rows]
            log_action("TOOL", f"Found {len(results)} patients.")
            return json.dumps(results)

        except Exception as e:
            return f"Error: {e}"
        finally:
            cur.close()
            close_connection(conn)

# --- TOOL 2: CONDITION ANALYZER ---
class ConditionInput(BaseModel):
    patient_ids: List[int] = Field(description="List of patient IDs")
    medical_concept: str = Field(description="Condition to search for (e.g., 'Asthma')")

class PatientConditionTool(BaseTool):
    name: str = "analyze_patient_condition"
    description: str = "Checks specific patients' notes for a condition using Vector Search."
    args_schema: Type[BaseModel] = ConditionInput

    def _run(self, patient_ids: List[int], medical_concept: str):
            if not patient_ids: return "No patient IDs provided."
            
            log_action("VECTOR", f"Embedding concept: '{medical_concept}'")
            query_vector = embed_fn.embed_query(medical_concept)
            query_vec_str = json.dumps(query_vector)

            conn = get_connection()
            cur = conn.cursor()
            try:
                placeholders = ','.join(['?'] * len(patient_ids))
                sql = f"""
                    SELECT p.id, p.full_name, cn.note_content, 
                        vector_distance_cos(cn.embedding, vector32(?)) as distance
                    FROM clinical_notes cn
                    JOIN patients p ON p.id = cn.patient_id
                    WHERE p.id IN ({placeholders})
                    ORDER BY distance ASC
                """
                params = [query_vec_str] + patient_ids
                
                log_action("SQL", f"Executing: SELECT ... FROM clinical_notes WHERE p.id IN ({patient_ids})")
                
                cur.execute(sql, params)
                rows = cur.fetchall()

                results = []
                for r in rows:
                    p_id, name, content, dist = r
                    
                    if dist < 0.5:
                        match_type = "HIGH CONFIDENCE" if dist < 0.35 else "LOW CONFIDENCE"
                        
                        # Log the specific distance so you can debug
                        log_action("VECTOR", f"Match: {name} | Dist: {dist:.3f} | Type: {match_type}")
                        
                        results.append({
                            "patient": f"{name} (ID: {p_id})",
                            "match_status": match_type, 
                            "semantic_distance": f"{dist:.3f}",
                            "note_excerpt": content[:150] + "..."
                        })

                if not results:
                    log_action("VECTOR", "No semantic matches found (All distances > 0.5).")
                    return f"No notes matched '{medical_concept}'."

                return json.dumps(results, indent=2)
            finally:
                cur.close()
                close_connection(conn)
                
# --- TOOL 3: DRUG SAFETY ---
class DrugSafetyInput(BaseModel):
    drug_name: str = Field(description="Name of the drug")

class DrugSafetyTool(BaseTool):
    name: str = "check_drug_safety"
    description: str = "Search Wiki for drug contraindications."
    args_schema: Type[BaseModel] = DrugSafetyInput

    def _run(self, drug_name: str):
            log_action("VECTOR", f"Searching Wiki for: {drug_name}")
            query_vector = embed_fn.embed_query(f"{drug_name} contraindications safety")
            query_vec_str = json.dumps(query_vector)
            
            conn = get_connection()
            cur = conn.cursor()
            try:
                sql = """
                    SELECT id, source_file, content, vector_distance_cos(embedding, vector32(?)) as distance
                    FROM medical_knowledge
                    ORDER BY distance ASC LIMIT 2
                """
                
                log_action("SQL", f"Executing Wiki Search: {sql.strip()}")
                
                cur.execute(sql, (query_vec_str,))
                rows = cur.fetchall()
                
                if not rows: return f"No info for {drug_name}."
                
                log_action("TOOL", f"Found {len(rows)} wiki articles.")
                
                results = []
                for r in rows:
                    chunk_id, source, content, dist = r
                    results.append(f"- [Source: {source} | ID: {chunk_id}] {content}")
                    print(f"\033[93m    -> {source} (ID: {chunk_id}): {content[:60]}...\033[0m")
                    
                return "\n\n".join(results)
            finally:
                cur.close()
                close_connection(conn)

# --- TOOL 4: IDENTITY ---
class IdentityInput(BaseModel):
    name_or_id: str = Field(description="Name or ID")

class IdentityTool(BaseTool):
    name: str = "lookup_patient_identity"
    description: str = "Find patient ID by name."
    args_schema: Type[BaseModel] = IdentityInput

    def _run(self, name_or_id: str):
        log_action("SQL", f"Looking up identity: {name_or_id}")
        conn = get_connection()
        cur = conn.cursor()
        try:
            cur.execute("SELECT id, full_name, system_id FROM patients WHERE full_name LIKE ? OR system_id = ?", 
                        (f"%{name_or_id}%", name_or_id))
            rows = cur.fetchall()
            return json.dumps([{"id": r[0], "name": r[1], "sys_id": r[2]} for r in rows])
        finally:
            cur.close()
            close_connection(conn)

# --- TOOL 5: SYSTEM STATS ---
class StatsInput(BaseModel):
    query_type: str = Field(description="Type of stat: 'files', 'chunks', 'total_patients'")

class SystemStatsTool(BaseTool):
    name: str = "get_system_stats"
    description: str = "Get meta-data: How many files ingested? How many chunks per file? Total patients?"
    args_schema: Type[BaseModel] = StatsInput

    def _run(self, query_type: str):
        log_action("TOOL", f"Fetching System Stats: {query_type}")
        conn = get_connection()
        cur = conn.cursor()
        try:
            if query_type == "files":
                cur.execute("SELECT filename, updated_at FROM ingestion_sources")
                rows = cur.fetchall()
                return f"Ingested Files:\n" + "\n".join([f"- {r[0]} (Updated: {r[1]})" for r in rows])
            
            elif query_type == "chunks":
                # Group by source_file to count chunks
                cur.execute("SELECT source_file, COUNT(*) FROM medical_knowledge GROUP BY source_file")
                rows = cur.fetchall()
                return f"Chunks per File:\n" + "\n".join([f"- {r[0]}: {r[1]} chunks" for r in rows])
            
            elif query_type == "total_patients":
                cur.execute("SELECT COUNT(*) FROM patients")
                count = cur.fetchone()[0]
                return f"Total Patients in DB: {count}"
            
            else:
                return "Unknown stat type. Use 'files', 'chunks', or 'total_patients'."
        finally:
            cur.close()
            close_connection(conn)
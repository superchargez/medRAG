# src/v3_core/tools.py
import json
from typing import Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from sqlalchemy import text, select
from langchain_core.prompts import PromptTemplate

# Imports
from src.v3_core.database import SessionLocal, Patient
from src.shared.config import EMBEDDING_API_URL
from src.shared.embedding import RemoteEmbeddingFunction
from src.shared.llm_factory import get_llm_chain

embed_fn = RemoteEmbeddingFunction(api_url=EMBEDDING_API_URL)

def get_db_session():
    return SessionLocal()

# --- 1. IDENTITY TOOL (Keep this: It is safer for exact ID lookups) ---
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

# --- 2. THE SUPER TOOL (Replaces PatientAnalyticsTool AND HistoryTool) ---
class SQLToolInput(BaseModel):
    question: str = Field(description="The full natural language question asking for data analysis, counts, history, or complex filtering.")

class MedicalSQLAgentTool(BaseTool):
    name: str = "medical_database_analyzer"
    description: str = (
        "Use this for ANY question about patients, medical history, counts, ages, dates, or conditions. "
        "Input: The full user question."
    )
    args_schema: Type[BaseModel] = SQLToolInput

    def _run(self, question: str):
        # 1. Define the Schema for the LLM
        # We include the vector function hint in the schema so the LLM knows it exists
        schema_desc = """
        Table: patients
        - id (integer, primary key)
        - system_id (text, e.g., 'PT-12345')
        - full_name (text)
        - dob (text, format 'YYYY-MM-DD')
        - gender (text)

        Table: clinical_notes
        - id (integer)
        - patient_id (integer, foreign key to patients.id)
        - visit_date (text, format 'YYYY-MM-DD')
        - condition (text, e.g., 'Flu', 'Hypertension')
        - note_content (text, medical notes)
        - embedding (vector column)
        """

        # 2. Create the Prompt
        prompt = PromptTemplate.from_template("""
        You are an expert SQLite/LibSQL Data Analyst.
        Given the database schema below, write a SQL query to answer the user's question.
        
        SCHEMA:
        {schema}

        RULES:
        1. Return ONLY the raw SQL query. No markdown.
        2. Dates are TEXT ('YYYY-MM-DD'). Use string comparison.
        3. Age Calculation: (strftime('%Y', 'now') - strftime('%Y', dob)).
        4. Text Search: Use `note_content LIKE '%term%'` for simple search.
        5. Vector Search: If the user asks for "semantic" or "meaning" search, you cannot do that here. Use LIKE for now.
        6. ALWAYS JOIN patients and clinical_notes when needed.
        
        QUESTION: {question}
        
        SQL QUERY:
        """)

        llm = get_llm_chain(provider="cerebras") 
        chain = prompt | llm
        
        try:
            generated_sql = chain.invoke({"schema": schema_desc, "question": question})
            clean_sql = generated_sql.content.replace("```sql", "").replace("```", "").strip()
            print(f"\n[SQL GENERATED]: {clean_sql}\n")
            
            db = get_db_session()
            try:
                # Execute
                result = db.execute(text(clean_sql)).fetchall()
                if not result: return "Query executed successfully but returned 0 results."
                return f"Query Results:\n{str(result)}"
            finally:
                db.close()
        except Exception as e:
            return f"Error generating SQL: {str(e)}"

# --- 3. WIKI TOOL (Keep for general knowledge) ---
class WikiInput(BaseModel):
    query: str = Field(description="Medical term or question.")

class MedicalWikiTool(BaseTool):
    name: str = "check_drug_info"
    description: str = "Search medical knowledge base (Wiki) using Vector Search."
    args_schema: Type[BaseModel] = WikiInput

    def _run(self, query: str):
        db = get_db_session()
        try:
            query_vec_str = json.dumps(embed_fn.embed_query(query))
            sql = text("""
                SELECT topic, content,
                       vector_distance_cos(embedding, vector32(:q)) as distance
                FROM medical_knowledge
                ORDER BY distance ASC
                LIMIT 3
            """)
            results = db.execute(sql, {"q": query_vec_str}).fetchall()
            if not results: return "No relevant medical info found."
            return "\n\n".join([f"[Topic: {r.topic}] {r.content}" for r in results])
        finally:
            db.close()
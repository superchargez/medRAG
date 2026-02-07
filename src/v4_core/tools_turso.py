# src/v4_core/tools_turso.py
import json
from typing import Type, Optional
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from src.shared.config import EMBEDDING_API_URL
from src.shared.embedding import RemoteEmbeddingFunction
from src.v4_core.database_turso import get_connection, close_connection, get_database_schema

# Initialize Embedding Function
embed_fn = RemoteEmbeddingFunction(api_url=EMBEDDING_API_URL)

# --- TOOL 1: SCHEMA INSPECTOR ---
class InspectSchemaInput(BaseModel):
    pass # No input needed

class InspectDatabaseTool(BaseTool):
    name: str = "inspect_database_schema"
    description: str = "Returns the list of tables and their columns (DDL). Use this FIRST to understand the database structure."
    args_schema: Type[BaseModel] = InspectSchemaInput

    def _run(self):
        return get_database_schema()

# --- TOOL 2: MEDICAL KNOWLEDGE (WIKI) ---
class WikiInput(BaseModel):
    query: str = Field(description="The medical concept to search for (e.g., 'Propranolol contraindications').")

class MedicalWikiTool(BaseTool):
    name: str = "search_medical_knowledge"
    description: str = "Search the medical wiki for drug interactions, disease definitions, or guidelines."
    args_schema: Type[BaseModel] = WikiInput

    def _run(self, query: str):
        query_vector = embed_fn.embed_query(query)
        query_vec_str = json.dumps(query_vector)
        
        conn = get_connection()
        cur = conn.cursor()
        try:
            # Semantic search on knowledge base
            cur.execute("""
                SELECT topic, content,
                       vector_distance_cos(embedding, vector32(?)) as distance
                FROM medical_knowledge
                ORDER BY distance ASC
                LIMIT 3
            """, (query_vec_str,))
            
            results = cur.fetchall()
            if not results:
                return "No relevant medical info found."
            
            response = f"--- Knowledge Results for '{query}' ---\n"
            for r in results:
                response += f"TOPIC: {r[0]}\nCONTENT: {r[1]}\n(Relevance: {1-r[2]:.2f})\n\n"
            return response
        finally:
            cur.close()
            close_connection(conn)

# --- TOOL 3: SEMANTIC SQL EXECUTOR ---
class SemanticSqlInput(BaseModel):
    sql_query: str = Field(description="The SQL query to execute. Must be valid SQLite/Turso syntax.")
    semantic_concept: Optional[str] = Field(
        default=None, 
        description="Optional. A medical concept (e.g., 'Asthma') to convert into a vector. If provided, use ':query_vector' in your SQL."
    )

class SemanticSQLTool(BaseTool):
    name: str = "execute_semantic_sql"
    description: str = (
        "Executes a SQL query against the database. "
        "POWERFUL FEATURE: If you need to filter by meaning (e.g., 'patients with heart issues'), "
        "provide the 'semantic_concept' argument (e.g., 'heart failure'). "
        "Then, in your SQL, use `vector_distance_cos(embedding, vector32(:query_vector))` to compare."
    )
    args_schema: Type[BaseModel] = SemanticSqlInput

    def _run(self, sql_query: str, semantic_concept: str = None):
        conn = get_connection()
        cur = conn.cursor()
        try:
            params = {}
            
            # If the agent wants to use vector search inside SQL
            if semantic_concept:
                print(f"   [Tool] Embedding concept: '{semantic_concept}'")
                vector = embed_fn.embed_query(semantic_concept)
                # Turso expects the vector as a JSON string for the vector32() function
                params['query_vector'] = json.dumps(vector)
            
            print(f"   [Tool] Executing SQL: {sql_query[:100]}...")
            
            # Execute query with parameters
            cur.execute(sql_query, params)
            rows = cur.fetchall()
            
            if not rows:
                return "Query executed successfully but returned 0 results."
            
            # Format results nicely
            # Get column names
            col_names = [description[0] for description in cur.description]
            
            result_str = f"Found {len(rows)} rows:\n"
            result_str += " | ".join(col_names) + "\n"
            result_str += "-" * 50 + "\n"
            
            # Limit return size to prevent context overflow
            MAX_ROWS = 20
            for i, row in enumerate(rows):
                if i >= MAX_ROWS:
                    result_str += f"\n... ({len(rows) - MAX_ROWS} more rows truncated)"
                    break
                result_str += str(row) + "\n"
                
            return result_str

        except Exception as e:
            return f"SQL Error: {str(e)}"
        finally:
            cur.close()
            close_connection(conn)
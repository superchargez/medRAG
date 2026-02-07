# src/v5_core/database.py
import turso
import os
from pathlib import Path

# Get the absolute path to the directory this file is in
CURRENT_DIR = Path(__file__).parent.resolve()
DB_FILE = CURRENT_DIR / "mediflow_turso_v5.db"

# Convert to string for the connector
DB_PATH = str(DB_FILE)

def get_connection():
    """Get a connection to the Turso database."""
    if not DB_FILE.exists():
        print(f"❌ CRITICAL ERROR: Database file not found at: {DB_PATH}")
    
    try:
        # Connect to the local file
        return turso.connect(DB_PATH)
    except Exception as e:
        print(f"❌ DATABASE CONNECTION FAILED: {e}")
        raise e

def get_cursor():
    """Get a cursor for the database."""
    conn = get_connection()
    return conn.cursor()

def close_connection(conn):
    """Close the database connection."""
    if conn:
        conn.close()

def get_database_schema() -> str:
    """Returns the DDL (CREATE TABLE statements)."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        tables = cur.fetchall()
        
        schema_text = "DATABASE SCHEMA:\n"
        for name, sql in tables:
            schema_text += f"--- TABLE: {name} ---\n"
            schema_text += f"{sql}\n\n"
            
        return schema_text
    except Exception as e:
        return f"Error retrieving schema: {e}"
    finally:
        cur.close()
        close_connection(conn)
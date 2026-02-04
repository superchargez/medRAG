# scripts/ingest_v3.py
import sys
import json
import random
from datetime import datetime, timedelta
from sqlalchemy import text
from faker import Faker
from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import traceback
import time

# Imports
from src.shared.config import EMBEDDING_API_URL
from src.shared.embedding import RemoteEmbeddingFunction
from src.v3_core.database import engine

# --- FIX FOR WINDOWS CONSOLE ENCODING ---
sys.stdout.reconfigure(encoding='utf-8')

def clean_text(text_content):
    if not text_content:
        return ""
    return text_content.encode('ascii', 'ignore').decode('ascii')

def calculate_dob(age):
    today = datetime.now()
    birth_date = today - timedelta(days=age * 365)
    return birth_date.date().isoformat()

def main():
    print("--- 🚀 STARTING TURSO MIGRATION (V3 - FIXED) ---")
    
    embed_fn = RemoteEmbeddingFunction(api_url=EMBEDDING_API_URL)
    fake = Faker()
    
    # Use connection context
    with engine.connect() as conn:
        try:
            # 1. SETUP DATABASE SCHEMA
            print("   > Setting up Database Schema...")
                        
            conn.execute(text("DROP TABLE IF EXISTS clinical_notes"))
            conn.execute(text("DROP TABLE IF EXISTS medical_knowledge"))
            conn.execute(text("DROP TABLE IF EXISTS patients"))
            
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS patients (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    system_id TEXT UNIQUE NOT NULL,
                    full_name TEXT NOT NULL,
                    gov_id TEXT UNIQUE NOT NULL,
                    phone TEXT,
                    dob TEXT NOT NULL,
                    gender TEXT
                )
            """))
            
            # Note: F32_BLOB(768) is the specific type for Turso vectors
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS clinical_notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id INTEGER NOT NULL,
                    visit_date TEXT NOT NULL,
                    condition TEXT NOT NULL,
                    note_content TEXT NOT NULL,
                    embedding F32_BLOB(768),
                    FOREIGN KEY (patient_id) REFERENCES patients(id)
                )
            """))
            
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS medical_knowledge (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding F32_BLOB(768)
                )
            """))
            
            print("   > Creating Indexes...")
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_patients_system_id ON patients(system_id)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_notes_patient_id ON clinical_notes(patient_id)"))
            
            # Vector Indexes 
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_notes_vector ON clinical_notes(libsql_vector_idx(embedding))"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_knowledge_vector ON medical_knowledge(libsql_vector_idx(embedding))"))
            
            conn.commit() # Commit schema changes
            print("   ✓ Database schema created")
            
            # --- PART A: MEDICAL KNOWLEDGE ---
            print("\n--- INGESTING MEDICAL KNOWLEDGE ---")
            topics = ["Asthma", "Hypertension", "Diabetes", "Lisinopril", "Aspirin"]
            
            for topic in topics:
                time.sleep(5) # Be polite to Wikipedia
                try:
                    print(f"   - Fetching: {topic}...")
                    loader = WikipediaLoader(query=topic, load_max_docs=1)
                    raw = loader.load()
                    if not raw: continue
                    
                    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                    chunks = splitter.split_documents(raw)
                    if not chunks: continue

                    texts_to_embed = [clean_text(doc.page_content) for doc in chunks]
                    print(f"     > Batch embedding {len(texts_to_embed)} chunks...")
                    vectors_list = embed_fn.embed_documents(texts_to_embed)
                    
                    for content, vector in zip(texts_to_embed, vectors_list):
                        vector_str = json.dumps(vector)
                        
                        conn.execute(
                            text("""
                                INSERT INTO medical_knowledge (topic, content, embedding)
                                VALUES (:topic, :content, vector32(:vec))
                            """),
                            {"topic": topic, "content": content, "vec": vector_str}
                        )
                    conn.commit() # Commit after each topic
                    print(f"     > Saved {len(chunks)} chunks.")
                        
                except Exception as e:
                    print(f"     ❌ Error fetching {topic}: {e}")
            
            # --- PART B: PATIENTS & NOTES ---
            print("\n--- INGESTING PATIENTS & NOTES ---")
            
            # 1. HERO PATIENT
            print("   > Adding hero patient...")
            conn.execute(
                text("""
                    INSERT INTO patients (system_id, full_name, gov_id, phone, dob, gender)
                    VALUES (:sid, :name, :gid, :phone, :dob, :gender)
                """),
                {
                    "sid": "PT-HERO-001", "name": "Ali Khan", "gid": "12345", 
                    "phone": "0300-1234567", "dob": calculate_dob(45), "gender": "Male"
                }
            )
            conn.commit()
            
            # Get ID
            result = conn.execute(text("SELECT id FROM patients WHERE system_id='PT-HERO-001'"))
            hero_id = result.scalar()
            
            # Hero Note
            note_text = "Patient complains of severe headache. BP 160/100. Diagnosis: Hypertension. He is also suffering from shortness of breath"
            vec_str = json.dumps(embed_fn.embed_query(note_text))
            
            conn.execute(
                text("""
                    INSERT INTO clinical_notes (patient_id, visit_date, condition, note_content, embedding)
                    VALUES (:pid, :date, :cond, :note, vector32(:vec))
                """),
                {
                    "pid": hero_id, "date": "2024-12-01", "cond": "Hypertension", 
                    "note": note_text, "vec": vec_str
                }
            )
            conn.commit()

            # 2. SYNTHETIC DATA
            print("   > Generating synthetic patients...")
            conditions = ["Asthma", "Hypertension", "Diabetes", "Flu"]
            genders = ["Male", "Female"]
            
            for i in range(30):
                gender = random.choice(genders)
                age = random.randint(18, 80)
                p_name = fake.name_male() if gender == "Male" else fake.name_female()
                sys_id = f"PT-{10000 + i:05d}"
                
                conn.execute(
                    text("""
                        INSERT INTO patients (system_id, full_name, gov_id, phone, dob, gender)
                        VALUES (:sid, :name, :gid, :phone, :dob, :gender)
                    """),
                    {
                        "sid": sys_id,
                        "name": p_name,
                        "gid": str(fake.random_number(digits=13)),
                        "phone": f"0300-{fake.random_number(digits=7)}",
                        "dob": calculate_dob(age),
                        "gender": gender
                    }
                )
                
                # We need to fetch the ID we just inserted to link the note
                # SQLite/LibSQL supports last_insert_rowid() but explicit select is safer in loops
                p_id_res = conn.execute(text("SELECT id FROM patients WHERE system_id=:sid"), {"sid": sys_id})
                p_id = p_id_res.scalar()
                
                cond = random.choice(conditions)
                n_text = f"Patient presented with {cond}. Age {age}."
                n_vec = json.dumps(embed_fn.embed_query(n_text))
                
                conn.execute(
                    text("""
                        INSERT INTO clinical_notes (patient_id, visit_date, condition, note_content, embedding)
                        VALUES (:pid, :date, :cond, :note, vector32(:vec))
                    """),
                    {
                        "pid": p_id,
                        "date": datetime.now().date().isoformat(),
                        "cond": cond,
                        "note": n_text,
                        "vec": n_vec
                    }
                )
                
                if (i + 1) % 10 == 0:
                    print(f"     > Processed {i + 1} patients...")
                    conn.commit() # Commit in batches
            
            conn.commit() # Final commit
            print("   > Patient Data Saved.")
            
        except Exception as e:
            print(f"   ❌ Error during ingestion: {e}")
            traceback.print_exc()
            # No need to rollback explicitly if using context manager without .begin(), 
            # but uncommitted changes won't persist if script crashes.
            raise
    
    # Verification
    print("\n--- VERIFYING INGESTION ---")
    with engine.connect() as conn:
        try:
            p_count = conn.execute(text("SELECT COUNT(*) FROM patients")).scalar()
            n_count = conn.execute(text("SELECT COUNT(*) FROM clinical_notes")).scalar()
            k_count = conn.execute(text("SELECT COUNT(*) FROM medical_knowledge")).scalar()
            
            print(f"   Total Patients: {p_count}")
            print(f"   Total Clinical Notes: {n_count}")
            print(f"   Total Medical Knowledge Entries: {k_count}")
            
            # Test Vector Search
            print("\n   > Testing Vector Search (Hypertension)...")
            q_vec = json.dumps(embed_fn.embed_query("High blood pressure"))
            
            # Syntax for vector search in Turso/LibSQL
            # Note: vector_top_k is often used as a function or index hint
            search_sql = """
                SELECT note_content, condition 
                FROM clinical_notes 
                ORDER BY vector_distance_cos(embedding, vector32(:vec)) ASC 
                LIMIT 1
            """
            res = conn.execute(text(search_sql), {"vec": q_vec}).fetchone()
            if res:
                print(f"     Match found: {res[1]} - {res[0][:50]}...")
            else:
                print("     ⚠ No vector match found (might need more data)")
            
        except Exception as e:
            print(f"   ❌ Verification error: {e}")

    print("\n--- ✅ TURSO INGESTION COMPLETE ---")

if __name__ == "__main__":
    main()
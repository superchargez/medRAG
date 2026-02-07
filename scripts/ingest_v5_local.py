# scripts/ingest_v5_local.py

import random
import time
from datetime import datetime, timedelta
from pathlib import Path
from faker import Faker
from langchain_text_splitters import RecursiveCharacterTextSplitter
import traceback
import hashlib
import turso
from src.shared.config import EMBEDDING_API_URL
from src.shared.embedding import RemoteEmbeddingFunction
from scripts.setup_turso_direct import MediFlowVectorIndex

DB_PATH = "src/v5_core/mediflow_turso_v5.db"
WIKI_DIR = Path("../wiki_knowledge")

def clean_text(text_content):
    if not text_content:
        return ""
    return text_content.encode('ascii', 'ignore').decode('ascii')

def calculate_dob(age):
    today = datetime.now()
    birth_date = today - timedelta(days=age * 365)
    return birth_date.date().isoformat()

def main():
    print("--- 🚀 STARTING BATCH INGESTION (V4) ---")
    print(f"Database: {DB_PATH}")
    print(f"Wiki Source: {WIKI_DIR.resolve()}")
    
    # Check if vector_indices exists and warn user about old data
    if Path("./vector_indices").exists():
        print("⚠️  WARNING: Existing 'vector_indices' folder found.")
        print("   If you encounter 'limit exceeded' errors, please delete this folder and restart.")
        time.sleep(2)
    
    embed_fn = RemoteEmbeddingFunction(api_url=EMBEDDING_API_URL)
    fake = Faker()
    vector_store = MediFlowVectorIndex()
    
    con = turso.connect(DB_PATH)
    cur = con.cursor()
    
    try:
        # Setup Patients Table
        print("\n   > Ensuring 'patients' table exists...")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                system_id TEXT UNIQUE NOT NULL,
                full_name TEXT NOT NULL,
                gov_id TEXT UNIQUE NOT NULL,
                phone TEXT,
                dob TEXT NOT NULL,
                gender TEXT
            )
        """)
        con.commit()
        
        # --- 1. PREPARE DATA & HASH CHECK ---
        print("\n📚 Preparing Data & Checking File Hashes...")
        if not WIKI_DIR.exists():
            print(f"❌ ERROR: Wiki directory not found at {WIKI_DIR}")
            return

        txt_files = sorted(WIKI_DIR.glob("*.txt"))
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        
        # Dictionary to hold files that NEED processing
        # key: filename, value: (full_content, file_hash)
        files_to_process = {}
        
        print("   > Checking for file changes...")
        
        for file_path in txt_files:
            filename = file_path.name
            try:
                # Read content once to calculate hash
                content_bytes = file_path.read_bytes()
                current_hash = hashlib.md5(content_bytes).hexdigest()
                content_str = content_bytes.decode('utf-8')
                
                # Check DB
                cur.execute("SELECT file_hash FROM ingestion_sources WHERE filename = ?", (filename,))
                row = cur.fetchone()
                
                should_process = False
                if row is None:
                    print(f"    [NEW] {filename}")
                    should_process = True
                elif row[0] != current_hash:
                    print(f"    [CHANGED] {filename}")
                    should_process = True
                else:
                    print(f"    [SKIP] {filename} (Unchanged)")
                
                if should_process:
                    files_to_process[filename] = {
                        'content': content_str,
                        'hash': current_hash
                    }
                    
            except Exception as e:
                print(f"    ! Warning: Could not process {filename}: {e}")

        if not files_to_process:
            print("   > No new or changed files found. Ingestion complete.")
        else:
            print(f"   > Found {len(files_to_process)} files to ingest/update.")

        # --- 2. BATCH INGEST KNOWLEDGE ---
        if files_to_process:
            print("\n💾 Processing Updates...")
            
            for filename, file_data in files_to_process.items():
                print(f"\n  > Processing: {filename}...")
                
                # 1. Purge old data for this file
                # This handles the "cascade" issue cleanly
                cur.execute("DELETE FROM medical_knowledge WHERE source_file = ?", (filename,))
                print(f"    - Purged old chunks for {filename}")
                
                # 2. Split new text
                chunks = splitter.split_text(file_data['content'])
                
                # 3. Prepare batch data: (topic, content, source_file)
                # We use the filename (without extension) as the topic for now
                topic = Path(filename).stem
                batch_data = [(topic, clean_text(chunk), filename) for chunk in chunks]
                
                # 4. Batch Upsert
                try:
                    vector_store.batch_upsert_knowledge(
                        con, cur, batch_data, embed_fn, auto_save=False
                    )
                    
                    # 5. Update the source tracker
                    cur.execute("""
                        INSERT OR REPLACE INTO ingestion_sources (filename, file_hash, updated_at)
                        VALUES (?, ?, ?)
                    """, (filename, file_data['hash'], datetime.now().isoformat()))
                    con.commit()
                    
                except Exception as e:
                    print(f"    ❌ Error processing {filename}: {e}")
                    con.rollback()
        
        # --- 3. INGEST PATIENTS (Standard) ---
        print("\n👥 Ingesting Patients...")
        # Hero Patient
        cur.execute("INSERT OR IGNORE INTO patients (system_id, full_name, gov_id, phone, dob, gender) VALUES (?, ?, ?, ?, ?, ?)",
                    ("PT-HERO-001", "Ali Khan", "12345-6789012-3", "0300-1234567", calculate_dob(45), "Male"))
        cur.execute("SELECT id FROM patients WHERE system_id = ?", ("PT-HERO-001",))
        hero_res = cur.fetchone()
        if hero_res:
            hero_id = hero_res[0]
            hero_note = "Patient complains of severe headache. BP 160/100. Diagnosis: Hypertension. He is also suffering from shortness of breath"
            # Using single insert for hero
            vector_store.upsert_clinical_note(con, cur, hero_id, "2024-12-01", "Hypertension", hero_note, embed_fn.embed_query, auto_save=False)

        # Synthetic Patients
        conditions = ["Asthma", "Hypertension", "Diabetes", "Flu"]
        for i in range(30):
            gender = random.choice(["Male", "Female"])
            age = random.randint(18, 80) # We still generate age for DOB calculation
            p_name = fake.name_male() if gender == "Male" else fake.name_female()
            sys_id = f"PT-{10000 + i:05d}"
            
            cur.execute("INSERT INTO patients (system_id, full_name, gov_id, phone, dob, gender) VALUES (?, ?, ?, ?, ?, ?)",
                        (sys_id, p_name, str(fake.random_number(digits=13)), f"0300-{fake.random_number(digits=7)}", calculate_dob(age), gender))
            
            p_id = cur.lastrowid
            cond = random.choice(conditions)
            visit_date = datetime.now().date().isoformat()
            n_text = n_text = f"Patient {p_name} (ID: {sys_id}) presented with {cond} on {visit_date}."
            
            vector_store.upsert_clinical_note(con, cur, p_id, visit_date, cond, n_text, embed_fn.embed_query, auto_save=False)
            
            if (i + 1) % 10 == 0:
                print(f"    > Processed {i + 1}/30 patients...")

        print("  ✓ Patient data ingestion complete.")
        
        # --- 4. FINALIZATION ---
        print("\n💾 Saving Indices to Disk...")
        vector_store.save_all_indices()
        con.commit()
        print("  ✓ Saved.")
        
        # --- 5. VERIFICATION ---
        print("\n✅ VERIFICATION")
        cur.execute("SELECT COUNT(*) FROM medical_knowledge")
        k_count = cur.fetchone()[0]
        print(f"  Total Knowledge: {k_count}")
        
        # Search Test
        print("\n  🧪 Testing Vector Search...")
        test_query = "High blood pressure"
        q_vec = embed_fn.embed_query(test_query)
        results = vector_store.search_similar_notes(q_vec, k=1)
        
        if results:
            # Assuming results is list of tuples (id, score)
            note_id = results[0][0]
            cur.execute("SELECT p.full_name, cn.condition FROM clinical_notes cn JOIN patients p ON p.id = cn.patient_id WHERE cn.id = ?", (note_id,))
            res = cur.fetchone()
            if res: print(f"    Match: {res[0]} ({res[1]})")
        
        print("\n🎉 COMPLETE!")
        
    except Exception as e:
        print(f"❌ Fatal Error: {e}")
        traceback.print_exc()
    finally:
        cur.close()
        con.close()

if __name__ == "__main__":
    main()
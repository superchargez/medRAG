# scripts/ingest_v2.py
import sys
import random
from datetime import datetime, timedelta
from sqlalchemy import text
from faker import Faker
from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Imports
from src.shared.config import EMBEDDING_API_URL
from src.shared.embedding import RemoteEmbeddingFunction
from src.shared.normalization import normalize_identifier
from src.v2_core.database import engine, Base, SessionLocal, Patient, ClinicalNote, MedicalKnowledge

# --- FIX FOR WINDOWS CONSOLE ENCODING ---
# This forces Python to ignore characters it can't print instead of crashing
sys.stdout.reconfigure(encoding='utf-8')

def clean_text(text):
    """
    Aggressively cleans text to prevent 'charmap' errors on Windows.
    """
    if not text: return ""
    # Encode to ASCII, ignore errors, then decode back. 
    # This strips emojis and weird math symbols.
    return text.encode('ascii', 'ignore').decode('ascii')

def calculate_dob(age):
    today = datetime.now()
    birth_date = today - timedelta(days=age*365)
    return birth_date.date()

def main():
    print("--- 🚀 STARTING SQL MIGRATION (V2 - GEMMA 768) ---")
    
    # 1. Initialize Database & Extension
    print("   > Setting up Database Schema...")
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    
    # Reset Tables
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

    embed_fn = RemoteEmbeddingFunction(api_url=EMBEDDING_API_URL)
    fake = Faker()
    db = SessionLocal()

    # --- PART A: MEDICAL KNOWLEDGE (WIKI) ---
    print("\n--- INGESTING MEDICAL KNOWLEDGE (SQL) ---")
    topics = ["Asthma", "Hypertension", "Diabetes", "Lisinopril", "Aspirin", "Metformin", "Influenza", "Pneumonia", "Chronic Obstructive Pulmonary Disease", "Coronary Artery Disease"]
    topics = ["Asthma", "Hypertension", "Diabetes", "Lisinopril", "Aspirin"]
    
    for topic in topics:
        try:
            print(f"   - Fetching: {topic}...")
            loader = WikipediaLoader(query=topic, load_max_docs=1)
            raw = loader.load()
            if not raw: continue
            
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(raw)
            
            for doc in chunks:
                # Clean text to avoid Windows errors
                safe_content = clean_text(doc.page_content)
                
                # Embed
                vector = embed_fn.embed_query(safe_content)
                
                # Save
                wiki_entry = MedicalKnowledge(
                    topic=topic,
                    content=safe_content,
                    embedding=vector
                )
                db.add(wiki_entry)
            
            db.commit() # Commit per topic
            print(f"     > Saved {len(chunks)} chunks for {topic}.")
            
        except Exception as e:
            db.rollback() # <--- CRITICAL FIX: Reset session on error
            print(f"     ❌ Error fetching {topic}: {e}")

    # --- PART B: PATIENTS & NOTES ---
    print("\n--- INGESTING PATIENTS & NOTES ---")

    try:
        # 1. HERO PATIENT
        hero = Patient(
            system_id="PT-HERO-001",
            full_name="Ali Khan",
            gov_id=normalize_identifier("12345"),
            phone=normalize_identifier("0300-1234567"),
            dob=calculate_dob(45),
            gender="Male"
        )
        db.add(hero)
        db.commit()

        # Hero Note
        note_text = "Patient complains of severe headache. BP 160/100. Diagnosis: Hypertension."
        note_1 = ClinicalNote(
            patient_id=hero.id,
            visit_date=datetime(2024, 12, 1).date(),
            condition="Hypertension",
            note_content=note_text,
            embedding=embed_fn.embed_query(note_text)
        )
        db.add(note_1)

        # 2. SYNTHETIC DATA
        conditions = ["Asthma", "Hypertension", "Diabetes", "Flu"]
        genders = ["Male", "Female"]

        for _ in range(30):
            gender = random.choice(genders)
            age = random.randint(18, 80)
            p_name = fake.name_male() if gender == "Male" else fake.name_female()
            
            p = Patient(
                system_id=f"PT-{fake.random_number(digits=5)}",
                full_name=p_name,
                gov_id=normalize_identifier(str(fake.random_number(digits=13))),
                phone=normalize_identifier(f"0300-{fake.random_number(digits=7)}"),
                dob=calculate_dob(age),
                gender=gender
            )
            db.add(p)
            db.commit()

            cond = random.choice(conditions)
            visit_date = (datetime.now() - timedelta(days=random.randint(0, 365))).date()
            n_text = f"Patient presented with {cond}. Age {age}."
            
            n = ClinicalNote(
                patient_id=p.id,
                visit_date=visit_date,
                condition=cond,
                note_content=n_text,
                embedding=embed_fn.embed_query(n_text)
            )
            db.add(n)
        
        db.commit()
        print("   > Patient Data Saved.")

    except Exception as e:
        db.rollback()
        print(f"   ❌ Error saving patients: {e}")

    db.close()
    print("\n--- ✅ SQL INGESTION COMPLETE ---")

if __name__ == "__main__":
    main()
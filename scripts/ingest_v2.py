# scripts/ingest_v2.py
import sys
import random
from datetime import datetime, timedelta
from sqlalchemy import select, func, text
from faker import Faker
from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time

# Imports
from src.shared.config import EMBEDDING_API_URL
from src.shared.embedding import RemoteEmbeddingFunction
from src.shared.normalization import normalize_identifier
from src.v2_core.database import engine, Base, SessionLocal, Patient, ClinicalNote, MedicalKnowledge

# --- FIX FOR WINDOWS CONSOLE ENCODING ---
sys.stdout.reconfigure(encoding='utf-8')

def clean_text(text):
    """
    Aggressively cleans text to prevent 'charmap' errors on Windows.
    """
    if not text: return ""
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
    
    # Create tables if they don't exist (DO NOT DROP)
    Base.metadata.create_all(engine)

    embed_fn = RemoteEmbeddingFunction(api_url=EMBEDDING_API_URL)
    fake = Faker()
    db = SessionLocal()

    # --- PART A: MEDICAL KNOWLEDGE (WIKI) ---
    print("\n--- INGESTING MEDICAL KNOWLEDGE (SQL) ---")
    topics = ["Asthma", "Hypertension", "Diabetes", "Lisinopril", "Aspirin", "Metformin"]
    # topics = ["Asthma", "Hypertension", "Diabetes", "Lisinopril", "Aspirin", "Metformin", "Influenza", "Pneumonia", "Chronic Obstructive Pulmonary Disease", "Coronary Artery Disease"]

    for topic in topics:
        try:
            # CHECK: Do we already have data for this topic?
            existing_count = db.scalar(
                select(func.count())
                .select_from(MedicalKnowledge)
                .where(MedicalKnowledge.topic == topic)
            )
            
            if existing_count and existing_count > 0:
                print(f"   - Skipping {topic} (already ingested).")
                continue

            # Fetch new data
            time.sleep(5) # Be polite to Wikipedia
            print(f"   - Fetching: {topic}...")
            
            loader = WikipediaLoader(query=topic, load_max_docs=1)
            raw = loader.load()
            
            if not raw: 
                print(f"     > No data found for {topic}, skipping.")
                continue
            
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(raw)
            
            count = 0
            for doc in chunks:
                safe_content = clean_text(doc.page_content)
                vector = embed_fn.embed_query(safe_content)
                
                wiki_entry = MedicalKnowledge(
                    topic=topic,
                    content=safe_content,
                    embedding=vector
                )
                db.add(wiki_entry)
                count += 1
            
            db.commit()
            print(f"     > Saved {count} chunks for {topic}.")
            
        except Exception as e:
            db.rollback()
            print(f"     ❌ Wiki Error ({topic}): {e}")
            print(f"     ⚠️  Creating dummy entry for testing...")
            
            # Fallback Dummy Entry
            dummy_text = f"{topic} is a medical condition. Dummy text."
            dummy_vec = embed_fn.embed_query(dummy_text)
            
            dummy_entry = MedicalKnowledge(
                topic=topic,
                content=dummy_text,
                embedding=dummy_vec
            )
            db.add(dummy_entry)
            db.commit()

    # --- PART B: PATIENTS & NOTES ---
    print("\n--- INGESTING PATIENTS & NOTES ---")

    try:
        # 1. HERO PATIENT (Check if exists)
        hero = db.scalar(select(Patient).where(Patient.system_id == "PT-HERO-001"))
        
        if not hero:
            print("Creating Hero Patient...")
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
        else:
            print("Hero Patient already exists. Skipping creation.")

        # 2. HERO NOTE (Check if exists)
        # We check if hero already has a note to prevent duplication
        hero_notes_count = db.scalar(
            select(func.count())
            .select_from(ClinicalNote)
            .where(ClinicalNote.patient_id == hero.id)
        )

        if hero_notes_count == 0:
            print("Adding Hero Note...")
            note_text = "Patient complains of severe headache. BP 160/100. Diagnosis: Hypertension."
            note_1 = ClinicalNote(
                patient_id=hero.id,
                visit_date=datetime(2024, 12, 1).date(),
                condition="Hypertension",
                note_content=note_text,
                embedding=embed_fn.embed_query(note_text)
            )
            db.add(note_1)
            db.commit()
        else:
            print("Hero Note already exists. Skipping creation.")

        # 3. SYNTHETIC DATA (Guard against infinite growth)
        # Only generate more patients if we have fewer than 50 total
        total_patients = db.scalar(select(func.count()).select_from(Patient))
        
        if total_patients < 50:
            print(f"Generating synthetic patients (Current: {total_patients})...")
            conditions = ["Asthma", "Hypertension", "Diabetes", "Flu"]
            genders = ["Male", "Female"]

            # Only generate enough to reach 50
            needed = 50 - total_patients
            
            for _ in range(needed):
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
            print("   > Synthetic Data Saved.")
        else:
            print(f"Skipping synthetic data generation (Sufficient data exists: {total_patients}).")

    except Exception as e:
        db.rollback()
        print(f"   ❌ Error saving patients: {e}")

    db.close()
    print("\n--- ✅ SQL INGESTION COMPLETE ---")

if __name__ == "__main__":
    main()
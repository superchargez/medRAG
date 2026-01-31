# scripts/ingest_v2.py
import chromadb
from langchain_chroma import Chroma
from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.indexing import index
from faker import Faker
from langchain_community.indexes._sql_record_manager import SQLRecordManager
import random
from datetime import datetime, timedelta

# --- IMPORTS ---
from src.shared.config import *
from src.shared.embedding import RemoteEmbeddingFunction
from src.shared.normalization import normalize_identifier

def calculate_dob(age):
    """Helper to generate a DOB based on an age."""
    today = datetime.now()
    birth_date = today - timedelta(days=age*365)
    return birth_date.strftime("%Y-%m-%d")

def main():
    print("--- 🚀 STARTING AGENTIC INGESTION (V2) ---")
    
    embed_fn = RemoteEmbeddingFunction(api_url=EMBEDDING_API_URL)
    chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

    # We use the SAME collection names, but we are overwriting/adding to them.
    # Ideally, for V2, you might want a fresh collection, but for now we update existing.
    patient_store = Chroma(
        client=chroma_client,
        collection_name=COLLECTION_PATIENTS,
        embedding_function=embed_fn,
        collection_metadata=COLLECTION_METADATA
    )

    rm_patients = SQLRecordManager(f"chroma/{COLLECTION_PATIENTS}", db_url=SQL_CONNECTION_STRING)
    rm_patients.create_schema()

    print("\n--- INGESTING ENRICHED PATIENT RECORDS ---")
    patient_docs = []
    fake = Faker()

    # 1. HERO PATIENT: Ali Khan (Enriched)
    hero_id = "PT-HERO-001"
    hero_name = "Ali Khan"
    hero_cnic = normalize_identifier("12345")
    hero_phone = normalize_identifier("0300-1234567")
    
    # Ali is 45 years old, Male
    ali_dob = calculate_dob(45) 

    patient_docs.append(Document(
        page_content=f"Patient: {hero_name}. DOB: {ali_dob}. Gender: Male. Date: 2024-12-01. Diagnosis: Hypertension. Rx: Lisinopril.",
        metadata={
            "source": f"{hero_id}_2024_v2", 
            "patient_id": hero_id, 
            "patient_name": hero_name,
            "gov_id": hero_cnic,
            "contact_number": hero_phone,
            "date": "2024-12-01", # Visit Date
            "dob": ali_dob,  
            "gender": "Male",
            "age": 45,
            "condition": "Hypertension",
            "type": "Vitals"
        }
    ))

    # 2. GENERATED PATIENTS (With Analytics Data)
    # We will generate 50 records to allow for "How many..." questions
    print("   > Generating 50 synthetic records...")
    
    conditions = ["Asthma", "Hypertension", "Diabetes", "Flu", "Migraine"]
    genders = ["Male", "Female"]
    
    for _ in range(50):
        pid = f"PT-{fake.random_number(digits=5)}"
        p_name = fake.name()
        gender = random.choice(genders)
        
        # Random Age between 18 and 80
        age = random.randint(18, 80)
        dob = calculate_dob(age)
        
        # Random Visit Date in the last 365 days
        days_ago = random.randint(0, 365)
        visit_date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        
        condition = random.choice(conditions)
        
        text = f"Patient: {p_name}. Gender: {gender}. Age: {age}. Date: {visit_date}. Diagnosis: {condition}."
        
        patient_docs.append(Document(
            page_content=text, 
            metadata={
                "source": pid, 
                "patient_id": pid, 
                "patient_name": p_name,
                "gov_id": normalize_identifier(str(fake.random_number(digits=13))),
                "contact_number": normalize_identifier(f"0300-{fake.random_number(digits=7)}"),
                "date": visit_date,
                "dob": dob,
                "gender": gender,
                "age": age,
                "condition": condition,
                "type": "General"
            }
        ))

    if patient_docs:
        # We use 'incremental' to add these new rich records alongside old ones
        # or 'full' if you want to wipe old data. Let's use incremental.
        index(
            patient_docs,
            rm_patients,
            patient_store,
            cleanup=INDEXING_CLEANUP,
            source_id_key="source",
            batch_size=INDEXING_BATCH_SIZE,
            key_encoder="sha256"
        )
        print("   > Enriched Patient Data Indexed.")

    print("\n--- ✅ V2 INGESTION COMPLETE ---")

if __name__ == "__main__":
    main()
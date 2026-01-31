# scripts/ingest_v1.py
import chromadb
from langchain_chroma import Chroma
from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.indexing import index
from faker import Faker
from langchain_community.indexes._sql_record_manager import SQLRecordManager

# --- IMPORTS FROM CONFIG ---
from src.shared.config import *
from src.shared.embedding import RemoteEmbeddingFunction
# NEW: Import normalization to ensure seed data matches API logic
from src.shared.normalization import normalize_identifier

def main():
    print("--- 🚀 STARTING INGESTION ---")
    
    # 1. Setup Clients
    embed_fn = RemoteEmbeddingFunction(api_url=EMBEDDING_API_URL)
    chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

    # 2. Setup Vector Stores
    wiki_store = Chroma(
        client=chroma_client,
        collection_name=COLLECTION_WIKI,
        embedding_function=embed_fn,
        collection_metadata=COLLECTION_METADATA
    )
    
    patient_store = Chroma(
        client=chroma_client,
        collection_name=COLLECTION_PATIENTS,
        embedding_function=embed_fn,
        collection_metadata=COLLECTION_METADATA
    )

    # 3. Setup Record Managers
    rm_wiki = SQLRecordManager(f"chroma/{COLLECTION_WIKI}", db_url=SQL_CONNECTION_STRING)
    rm_patients = SQLRecordManager(f"chroma/{COLLECTION_PATIENTS}", db_url=SQL_CONNECTION_STRING)
    
    rm_wiki.create_schema()
    rm_patients.create_schema()

    # --- PART A: WIKI DATA (Medical Knowledge) ---
    print("\n--- INGESTING MEDICAL KNOWLEDGE ---")
    topics = ["Propranolol", "Beta blocker", "Asthma", "Hypertension", "Drug interaction", "Headache", "Aspirin", "Bradycardia", "Migraine"]
    wiki_docs = []
    
    for topic in topics:
        try:
            print(f"   - Fetching: {topic}...")
            loader = WikipediaLoader(query=topic, load_max_docs=1)
            raw = loader.load()
            if not raw:
                continue
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(raw)
            for doc in chunks:
                doc.metadata["source"] = f"wiki_{topic}" 
                doc.metadata["topic"] = topic
            wiki_docs.extend(chunks)
        except Exception as e:
            print(f"     Error fetching {topic}: {e}")

    if wiki_docs:
        index(wiki_docs, rm_wiki, wiki_store, cleanup=INDEXING_CLEANUP, source_id_key="source", batch_size=INDEXING_BATCH_SIZE, key_encoder="sha256")
        print("   > Wiki Data Indexed.")

    # --- PART B: PATIENT DATA (Updated with New Schema) ---
    print("\n--- INGESTING PATIENT RECORDS ---")
    patient_docs = []
    
    # 1. HERO PATIENT: Ali Khan (The one we test with)
    hero_id = "PT-HERO-001"
    hero_name = "Ali Khan"
    # We set specific numbers for testing the new identity logic
    hero_cnic = "12345"
    hero_phone = "0300-1234567"
    
    clean_hero_cnic = normalize_identifier(hero_cnic)
    clean_hero_phone = normalize_identifier(hero_phone)
    
    # Record 1: 2020 (Asthma)
    patient_docs.append(Document(
        page_content=f"Patient: {hero_name}. Date: 2020-05-10. Diagnosis: Severe Bronchial Asthma. Rx: Salbutamol Inhaler. ID: {hero_id}.",
        metadata={
            "source": f"{hero_id}_2020", 
            "patient_id": hero_id, 
            "patient_name": hero_name,
            "gov_id": clean_hero_cnic,
            "contact_number": clean_hero_phone,
            "date": "2020-05-10",
            "condition": "Asthma",
            "type": "Diagnosis"
        }
    ))
    
    # Record 2: 2024 (Hypertension)
    patient_docs.append(Document(
        page_content=f"Patient: {hero_name}. Date: 2024-12-01. Vitals: BP 160/100. Diagnosis: Hypertension. Rx: Lisinopril. Phone: {hero_phone}.",
        metadata={
            "source": f"{hero_id}_2024", 
            "patient_id": hero_id, 
            "patient_name": hero_name,
            "gov_id": clean_hero_cnic,
            "contact_number": clean_hero_phone,
            "date": "2024-12-01",
            "condition": "Hypertension",
            "type": "Vitals"
        }
    ))

    # 2. FAMILY MEMBER: Sara Khan (Shares Phone, different ID/CNIC)
    sara_id = "PT-SARA-002"
    sara_cnic = normalize_identifier("99999")
    
    patient_docs.append(Document(
        page_content=f"Patient: Sara Khan. Date: 2023-08-15. Diagnosis: Migraine. Phone: {hero_phone}.",
        metadata={
            "source": f"{sara_id}_2023",
            "patient_id": sara_id,
            "patient_name": "Sara Khan",
            "gov_id": sara_cnic,
            "contact_number": clean_hero_phone,
            "date": "2023-08-15",
            "condition": "Migraine",
            "type": "General"
        }
    ))

    # 3. GENERATED PATIENTS
    fake = Faker()
    for _ in range(10):
        pid = f"PT-{fake.random_number(digits=5)}"
        p_name = fake.name()
        
        # Generate random identifiers
        # fake.ssn() usually creates dash-separated numbers, perfect for testing normalization
        rand_cnic = normalize_identifier(str(fake.random_number(digits=13)))
        rand_phone = normalize_identifier(f"0300-{fake.random_number(digits=7)}")
        
        is_asthmatic = fake.boolean(chance_of_getting_true=30)
        condition = "Asthma" if is_asthmatic else "Hypertension"
        rx = "Salbutamol" if is_asthmatic else "Lisinopril"
        
        text = f"Patient: {p_name}. Diagnosis: {condition}. Rx: {rx}."
        
        patient_docs.append(Document(
            page_content=text, 
            metadata={
                "source": pid, 
                "patient_id": pid, 
                "patient_name": p_name,
                "gov_id": rand_cnic,
                "contact_number": rand_phone,
                "type": "General",
                "condition": condition
            }
        ))

    if patient_docs:
        indexing_patients = index(
            patient_docs,
            rm_patients,
            patient_store,
            cleanup=INDEXING_CLEANUP,
            source_id_key="source",
            batch_size=INDEXING_BATCH_SIZE,
            key_encoder="sha256"
        )
        print(f"   > Patient Indexing Stats: {indexing_patients}")

    print("\n--- ✅ SYSTEM READY ---")

if __name__ == "__main__":
    main()
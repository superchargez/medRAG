# src/core/retriever.py
from langchain_chroma import Chroma

def get_wiki_context(terms: list, wiki_store: Chroma) -> str:
    if not terms:
        return "No specific medical terms identified."
    query = " ".join(terms)
    docs = wiki_store.similarity_search(query, k=3)
    return "\n\n".join([f"[Source: {d.metadata.get('topic', 'Wiki')}] {d.page_content}" for d in docs])

def get_patient_context(patient_id: str, query_text: str, patient_store: Chroma) -> str:
    """
    Fetches patient records relevant to the SPECIFIC query.
    """
    
    docs = patient_store.similarity_search(
        query_text, 
        k=15, 
        filter={"patient_id": patient_id}
    )
    
    if not docs:
        return "No records found for this patient ID."

    context_parts = []
    for d in docs:
        date = d.metadata.get("date", "Unknown Date")
        source = d.metadata.get("type", "Record")
        context_parts.append(f"Date: {date} | Type: {source} | Note: {d.page_content}")
        
    return "\n".join(context_parts)
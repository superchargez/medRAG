# src/shared/wiki_.py
from langchain_chroma import Chroma

def get_wiki_context(terms: list, wiki_store: Chroma) -> str:
    if not terms:
        return "No specific medical terms identified."
    query = " ".join(terms)
    docs = wiki_store.similarity_search(query, k=3)
    return "\n\n".join([f"[Source: {d.metadata.get('topic', 'Wiki')}] {d.page_content}" for d in docs])

# src/v2_core/database.py
from sqlalchemy import create_engine, Integer, String, Date, Text, ForeignKey
from sqlalchemy.orm import DeclarativeBase, relationship, sessionmaker, Mapped, mapped_column
from pgvector.sqlalchemy import Vector
from src.shared.config import POSTGRES_DB_URL

# 1. Setup Engine
engine = create_engine(POSTGRES_DB_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
dim=768

class Base(DeclarativeBase):
    pass

# 2. Define Models

class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    system_id: Mapped[str] = mapped_column(String, unique=True, index=True) # PT-XXX
    full_name: Mapped[str] = mapped_column(String, index=True)
    gov_id: Mapped[str] = mapped_column(String, unique=True, index=True)
    phone: Mapped[str] = mapped_column(String)
    dob: Mapped[str] = mapped_column(Date)
    gender: Mapped[str] = mapped_column(String)

    notes = relationship("ClinicalNote", back_populates="patient")

class ClinicalNote(Base):
    __tablename__ = "clinical_notes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"))
    
    visit_date: Mapped[str] = mapped_column(Date, index=True)
    condition: Mapped[str] = mapped_column(String, index=True)
    note_content: Mapped[str] = mapped_column(Text)
    
    # 768 dimensions (Standard for embeddinggema). 384 for minilm.
    embedding = mapped_column(Vector(dim))

    patient = relationship("Patient", back_populates="notes")

class MedicalKnowledge(Base):
    """
    Replaces Chroma 'wiki' collection.
    Stores chunks of medical text for RAG.
    """
    __tablename__ = "medical_knowledge"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    topic: Mapped[str] = mapped_column(String, index=True) # e.g. "Asthma"
    content: Mapped[str] = mapped_column(Text)             # The chunk text
    embedding = mapped_column(Vector(dim))                 # The vector

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
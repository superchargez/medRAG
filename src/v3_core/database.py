# src/v3_core/database.py
from sqlalchemy import create_engine, Integer, String, Text, ForeignKey
from sqlalchemy.orm import DeclarativeBase, relationship, sessionmaker, Mapped, mapped_column
from src.shared.config import TURSO_DB_URL
import os
from sqlalchemy.pool import NullPool

# Ensure we have the libsql dialect registered
try:
    # import libsql_experimental
    print("✓ libsql-experimental dialect registered")
except ImportError:
    print("⚠ libsql-experimental not installed. Run: pip install libsql-experimental")

if "sqlite+libsql" in TURSO_DB_URL:
    DATABASE_URL = TURSO_DB_URL
elif TURSO_DB_URL.startswith("http"):
    clean_url = TURSO_DB_URL.replace("http://", "").replace("https://", "")
    DATABASE_URL = f"sqlite+libsql://{clean_url}"
else:
    # Default fallback for local dev
    DATABASE_URL = "sqlite+libsql://127.0.0.1:8008"

print(f"Database URL: {DATABASE_URL}")

# 1. Setup Engine (LibSQL)
engine = create_engine(
    DATABASE_URL,
    echo=False,  # Set to False to reduce noise during ingestion
    poolclass=NullPool,  # Disable pooling for serverless DBs
    pool_pre_ping=True
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Base(DeclarativeBase):
    pass

# 2. Define Models
class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    system_id: Mapped[str] = mapped_column(String, unique=True, index=True) 
    full_name: Mapped[str] = mapped_column(String, index=True)
    gov_id: Mapped[str] = mapped_column(String, unique=True, index=True)
    phone: Mapped[str] = mapped_column(String)
    dob: Mapped[str] = mapped_column(String)
    gender: Mapped[str] = mapped_column(String)

    notes = relationship("ClinicalNote", back_populates="patient", cascade="all, delete-orphan")

class ClinicalNote(Base):
    __tablename__ = "clinical_notes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id", ondelete="CASCADE"))
    
    visit_date: Mapped[str] = mapped_column(String, index=True)
    condition: Mapped[str] = mapped_column(String, index=True)
    note_content: Mapped[str] = mapped_column(Text)
    
    # Embedding handled via raw SQL
    
    patient = relationship("Patient", back_populates="notes")

class MedicalKnowledge(Base):
    __tablename__ = "medical_knowledge"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    topic: Mapped[str] = mapped_column(String, index=True)
    content: Mapped[str] = mapped_column(Text)
    
    # Embedding handled via raw SQL

def get_db():
    """Dependency to get DB session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
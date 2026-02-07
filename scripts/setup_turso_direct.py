# scripts/setup_turso_direct.py
import sys
import os
import json
import hashlib
import hnswlib
import numpy as np
from typing import Optional, Callable, Any, List, Tuple
import turso

# Database path
DB_PATH = "src/v4_core/mediflow_turso_v4.db"

class MediFlowVectorIndex:
    """
    A class to manage vector embeddings for MediFlow using hnswlib for local vector indexing.
    This handles both database operations and vector index management.
    """
    
    def __init__(self, index_dir: str = "./vector_indices"):
        """
        Initialize the MediFlowVectorIndex.
        
        Args:
            index_dir: Directory to store hnswlib index files
        """
        self.index_dir = index_dir
        os.makedirs(index_dir, exist_ok=True)
        
        # Initialize hnswlib indices
        self.notes_index = None
        self.knowledge_index = None
        self.notes_labels = {}  # Map from database ID to hnswlib label
        self.knowledge_labels = {}  # Map from database ID to hnswlib label
        self.next_notes_label = 0
        self.next_knowledge_label = 0

        # Load Notes Map
        notes_map_path = os.path.join(self.index_dir, "notes_labels.json")
        if os.path.exists(notes_map_path):
            with open(notes_map_path, 'r') as f:
                # Convert keys back to int because JSON converts them to strings
                self.notes_labels = {int(k): v for k, v in json.load(f).items()}
                
        # Load Knowledge Map
        knowledge_map_path = os.path.join(self.index_dir, "knowledge_labels.json")
        if os.path.exists(knowledge_map_path):
            with open(knowledge_map_path, 'r') as f:
                self.knowledge_labels = {int(k): v for k, v in json.load(f).items()}

        # Load or create indices
        self._init_indices()
    
    def _init_indices(self):
        """Initialize or load existing hnswlib indices with auto-recovery."""
        # Increased default limit to avoid frequent rebuilds
        MAX_ELEMENTS = 100000 
        
        # --- Helper to handle the logic for a single index ---
        def setup_index(index_type, index_path):
            index = hnswlib.Index(space='cosine', dim=768)
            labels = {}
            next_label = 0
            needs_rebuild = False

            if os.path.exists(index_path):
                try:
                    index.load_index(index_path)
                    next_label = index.get_current_count()
                    print(f"Loaded {index_type} index with {next_label} vectors")
                except RuntimeError as e:
                    # CATCH THE ERROR: "The number of elements exceeds..."
                    print(f"⚠️  CRITICAL: {index_type} index corrupted or limit reached: {e}")
                    print(f"    🔄 Initiating automatic rebuild...")
                    needs_rebuild = True
            
            if not os.path.exists(index_path) or needs_rebuild:
                if needs_rebuild:
                    # Delete the old files to force a fresh start
                    try:
                        os.remove(index_path)
                        label_path = index_path.replace(".bin", "_labels.json")
                        if os.path.exists(label_path):
                            os.remove(label_path)
                        print(f"    🗑️  Cleaned up old {index_type} files.")
                    except:
                        pass
                
                print(f"    🆕 Creating new {index_type} index (Max Elements: {MAX_ELEMENTS})...")
                index.init_index(max_elements=MAX_ELEMENTS, ef_construction=200, M=16)
                
                # REBUILD LOGIC: Reload data from SQL
                labels, next_label = self._rebuild_from_sql(index_type)
            
            return index, labels, next_label

        # --- Execute for Notes ---
        notes_index_path = os.path.join(self.index_dir, "notes_index.bin")
        self.notes_index, self.notes_labels, self.next_notes_label = setup_index("notes", notes_index_path)

        # --- Execute for Knowledge ---
        knowledge_index_path = os.path.join(self.index_dir, "knowledge_index.bin")
        self.knowledge_index, self.knowledge_labels, self.next_knowledge_label = setup_index("knowledge", knowledge_index_path)

    def _rebuild_from_sql(self, index_type):
        """
        Recreates the vector index by reading existing embeddings from the database.
        This avoids re-embedding the text.
        """
        print(f"    📥 Rebuilding {index_type} index from database (this may take a moment)...")
        
        # We need a temporary connection to fetch data since we don't pass conn/cur in __init__
        # Note: In a strict MVC design this is bad practice, but for this script it's necessary for recovery.
        import turso
        con = turso.connect(DB_PATH)
        cur = con.cursor()
        
        labels = {}
        next_label = 0
        count = 0
        
        # Determine table and label map
        if index_type == "notes":
            table = "clinical_notes"
            label_map = self.notes_labels
            index_obj = self.notes_index
        else:
            table = "medical_knowledge"
            label_map = self.knowledge_labels
            index_obj = self.knowledge_index
            
        try:
            # Fetch all ID and Embeddings
            cur.execute(f"SELECT id, embedding FROM {table}")
            rows = cur.fetchall()
            
            if not rows:
                print(f"    ℹ️  No data found in {table} to rebuild.")
                return {}, 0
            
            # Prepare batches for HNSW
            ids = []
            vectors = []
            
            for db_id, embedding_blob in rows:
                # Decode blob (it's stored as JSON string in F32_BLOB)
                try:
                    # If the blob is bytes, decode to str first
                    if isinstance(embedding_blob, bytes):
                        embedding_blob = embedding_blob.decode('utf-8')
                    vector = json.loads(embedding_blob)
                    
                    ids.append(db_id)
                    vectors.append(vector)
                    
                    # Track labels
                    labels[db_id] = next_label
                    next_label += 1
                    count += 1
                    
                except Exception as e:
                    print(f"    ! Warning: Skipping ID {db_id} due to decode error: {e}")
            
            # Add all vectors to HNSW at once (Efficient)
            if vectors:
                np_vectors = np.array(vectors, dtype=np.float32)
                np_labels = np.arange(next_label - len(vectors), next_label)
                index_obj.add_items(np_vectors, np_labels)
                
            print(f"    ✅ Rebuild Complete: Added {count} vectors to {index_type} index.")
            
        finally:
            cur.close()
            con.close()
            
        return labels, next_label
    
    def _generate_content_hash(self, content: str) -> str:
        """Generate SHA256 hash for content."""
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _embedding_to_blob(self, embedding: List[float]) -> str:
        """Convert embedding list to JSON string for storage."""
        return json.dumps(embedding)
    
    def _blob_to_embedding(self, blob: str) -> List[float]:
        """Convert stored blob to embedding list."""
        return json.loads(blob)

    def batch_upsert_knowledge(self, conn: Any, cur: Any, data: List[Tuple[str, str, str]], 
                            embedding_func, auto_save: bool = False, batch_size: int = 32):
        """
        Batch upsert medical knowledge with controlled batch size.
        
        Args:
            data: List of tuples (topic, content, source_file)
            embedding_func: The embedding object
            batch_size: Number of documents to embed per API call (default: 32)
        """
        if not data:
            return

        print(f"    > Processing {len(data)} items in batches of {batch_size}...")
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(data), batch_size):
            batch_end = min(i + batch_size, len(data))
            batch_data = data[i:batch_end]
            batch_contents = [item[1] for item in batch_data]
            
            print(f"    > Embedding batch {i//batch_size + 1}/{(len(data)-1)//batch_size + 1} ({len(batch_contents)} items)...")
            
            try:
                batch_embeddings = embedding_func.embed_documents(batch_contents)
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"    ❌ Error in batch {i//batch_size + 1}: {e}")
                # Fill with zeros if batch fails (or handle differently)
                all_embeddings.extend([[0.0] * 768] * len(batch_contents))
        
        # 2. Prepare data for DB insertion (same as before)
        topics = [item[0] for item in data]
        contents = [item[1] for item in data]
        source_files = [item[2] for item in data]
        
        rows_to_insert = []
        hash_map = {} 
        
        for i, (topic, content, embedding) in enumerate(zip(topics, contents, all_embeddings)):
            content_hash = self._generate_content_hash(content)
            embedding_blob = self._embedding_to_blob(embedding)
            src_file = source_files[i]
            
            rows_to_insert.append((
                topic, content, embedding_blob, content_hash, src_file
            ))
            hash_map[content_hash] = i
            
        # 3. Batch Insert (rest remains the same)
        print(f"    > Inserting {len(rows_to_insert)} rows into DB...")
        cur.executemany("""
            INSERT OR IGNORE INTO medical_knowledge 
            (topic, content, embedding, content_hash, source_file)
            VALUES (?, ?, vector32(?), ?, ?)
        """, rows_to_insert)
        
        conn.commit()
        
        # 4. HNSW index update (same as before)
        placeholders = ','.join(['?'] * len(hash_map))
        cur.execute(f"""
            SELECT id, content_hash FROM medical_knowledge 
            WHERE content_hash IN ({placeholders})
        """, list(hash_map.keys()))
        
        new_items_for_hnsw = []
        labels_for_hnsw = []
        
        for db_id, content_hash in cur.fetchall():
            if db_id not in self.knowledge_labels:
                original_index = hash_map[content_hash]
                vector = all_embeddings[original_index]
                
                new_items_for_hnsw.append(vector)
                labels_for_hnsw.append(self.next_knowledge_label)
                
                self.knowledge_labels[db_id] = self.next_knowledge_label
                self.next_knowledge_label += 1
        
        # 5. Add to HNSW
        if new_items_for_hnsw:
            print(f"    > Adding {len(new_items_for_hnsw)} vectors to HNSW Index...")
            np_vectors = np.array(new_items_for_hnsw, dtype=np.float32)
            self.knowledge_index.add_items(np_vectors, labels_for_hnsw)
            
            if auto_save:
                self.save_knowledge_index()

    def upsert_clinical_note(self, conn: Any, cur: Any, patient_id: int, 
                           visit_date: str, condition: str, note_content: str,
                           embedding_func: Callable[[str], List[float]],
                           auto_save: bool = True) -> int:
        """
        Upsert a clinical note with embedding and vector index update.
        
        Args:
            conn: Database connection
            cur: Database cursor
            patient_id: Patient ID
            visit_date: Visit date string
            condition: Medical condition
            note_content: Clinical note content
            embedding_func: Function that takes text and returns embedding
            auto_save: Whether to automatically save the index after update
            
        Returns:
            The ID of the inserted/updated record
        """
        try:
            # Generate embedding
            embedding = embedding_func(note_content)
            embedding_blob = self._embedding_to_blob(embedding)
            
            # Generate content hash for deduplication
            content_hash = self._generate_content_hash(note_content)
            
            # Check if record exists by content hash
            cur.execute("""
                SELECT id, embedding FROM clinical_notes 
                WHERE content_hash = ?
            """, (content_hash,))
            
            existing = cur.fetchone()
            
            if existing:
                # Update existing record
                record_id = existing[0]
                existing_embedding = existing[1]
                
                # Update database record
                cur.execute("""
                    UPDATE clinical_notes 
                    SET patient_id = ?, visit_date = ?, condition = ?, 
                        note_content = ?, embedding = vector32(?)
                    WHERE id = ?
                """, (patient_id, visit_date, condition, note_content, 
                      embedding_blob, record_id))
                
                # Update vector index if embedding changed
                if existing_embedding != embedding_blob:
                    if record_id in self.notes_labels:
                        # Remove old vector and add new one with same label
                        label = self.notes_labels[record_id]
                        # Note: hnswlib doesn't support direct update, so we mark as deleted
                        # In practice, we might need a different approach for updates
                        embedding_array = np.array([embedding], dtype=np.float32)
                        self.notes_index.add_items(embedding_array, [label])
                
            else:
                # Insert new record
                cur.execute("""
                    INSERT INTO clinical_notes 
                    (patient_id, visit_date, condition, note_content, 
                     embedding, content_hash)
                    VALUES (?, ?, ?, ?, vector32(?), ?)
                """, (patient_id, visit_date, condition, note_content, 
                      embedding_blob, content_hash))
                
                record_id = cur.lastrowid
                
                # Add to vector index
                embedding_array = np.array([embedding], dtype=np.float32)
                label = self.next_notes_label
                self.notes_index.add_items(embedding_array, [label])
                
                # Update mappings
                self.notes_labels[record_id] = label
                self.next_notes_label += 1
            
            conn.commit()
            
            # Save index if requested
            if auto_save:
                self.save_notes_index()
            
            return record_id
            
        except Exception as e:
            conn.rollback()
            raise Exception(f"Error upserting clinical note: {e}")
    
    def upsert_knowledge(self, conn: Any, cur: Any, topic: str, content: str,
                        embedding_func: Callable[[str], List[float]],
                        auto_save: bool = True) -> int:
        """
        Upsert medical knowledge with embedding and vector index update.
        
        Args:
            conn: Database connection
            cur: Database cursor
            topic: Knowledge topic
            content: Knowledge content
            embedding_func: Function that takes text and returns embedding
            auto_save: Whether to automatically save the index after update
            
        Returns:
            The ID of the inserted/updated record
        """
        try:
            # Generate embedding
            embedding = embedding_func(content)
            embedding_blob = self._embedding_to_blob(embedding)
            
            # Generate content hash for deduplication
            content_hash = self._generate_content_hash(content)
            
            # Check if record exists by content hash
            cur.execute("""
                SELECT id, embedding FROM medical_knowledge 
                WHERE content_hash = ?
            """, (content_hash,))
            
            existing = cur.fetchone()
            
            if existing:
                # Update existing record
                record_id = existing[0]
                existing_embedding = existing[1]
                
                # Update database record
                cur.execute("""
                    UPDATE medical_knowledge 
                    SET topic = ?, content = ?, embedding = vector32(?)
                    WHERE id = ?
                """, (topic, content, embedding_blob, record_id))
                
                # Update vector index if embedding changed
                if existing_embedding != embedding_blob:
                    if record_id in self.knowledge_labels:
                        label = self.knowledge_labels[record_id]
                        embedding_array = np.array([embedding], dtype=np.float32)
                        self.knowledge_index.add_items(embedding_array, [label])
                
            else:
                # Insert new record
                cur.execute("""
                    INSERT INTO medical_knowledge 
                    (topic, content, embedding, content_hash)
                    VALUES (?, ?, vector32(?), ?)
                """, (topic, content, embedding_blob, content_hash))
                
                record_id = cur.lastrowid
                
                # Add to vector index
                embedding_array = np.array([embedding], dtype=np.float32)
                label = self.next_knowledge_label
                self.knowledge_index.add_items(embedding_array, [label])
                
                # Update mappings
                self.knowledge_labels[record_id] = label
                self.next_knowledge_label += 1
            
            conn.commit()
            
            # Save index if requested
            if auto_save:
                self.save_knowledge_index()
            
            return record_id
            
        except Exception as e:
            conn.rollback()
            raise Exception(f"Error upserting medical knowledge: {e}")
    
    def search_similar_notes(self, query_embedding: List[float], k: int = 5) -> List[Tuple[int, float]]:
        """
        Search for similar clinical notes.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of (database_id, similarity_score) tuples
        """
        query_array = np.array([query_embedding], dtype=np.float32)
        labels, distances = self.notes_index.knn_query(query_array, k=k)
        
        # Convert hnswlib labels back to database IDs
        results = []
        label_to_id = {v: k for k, v in self.notes_labels.items()}
        
        for label, distance in zip(labels[0], distances[0]):
            db_id = label_to_id.get(label)
            if db_id:
                # Convert distance to similarity score (1 - distance for cosine)
                similarity = 1.0 - distance
                results.append((db_id, similarity))
        
        return results
    
    def search_similar_knowledge(self, query_embedding: List[float], k: int = 5) -> List[Tuple[int, float]]:
        """
        Search for similar medical knowledge.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of (database_id, similarity_score) tuples
        """
        query_array = np.array([query_embedding], dtype=np.float32)
        labels, distances = self.knowledge_index.knn_query(query_array, k=k)
        
        # Convert hnswlib labels back to database IDs
        results = []
        label_to_id = {v: k for k, v in self.knowledge_labels.items()}
        
        for label, distance in zip(labels[0], distances[0]):
            db_id = label_to_id.get(label)
            if db_id:
                similarity = 1.0 - distance
                results.append((db_id, similarity))
        
        return results
    
    def save_notes_index(self):
        """Save the clinical notes index and labels to disk."""
        index_path = os.path.join(self.index_dir, "notes_index.bin")
        self.notes_index.save_index(index_path)
        
        # Save the label map
        map_path = os.path.join(self.index_dir, "notes_labels.json")
        with open(map_path, 'w') as f:
            json.dump(self.notes_labels, f)

    def save_knowledge_index(self):
        """Save the medical knowledge index and labels to disk."""
        index_path = os.path.join(self.index_dir, "knowledge_index.bin")
        self.knowledge_index.save_index(index_path)
        
        map_path = os.path.join(self.index_dir, "knowledge_labels.json")
        with open(map_path, 'w') as f:
            json.dump(self.knowledge_labels, f)    

    def save_all_indices(self):
        """Save all indices to disk."""
        self.save_notes_index()
        self.save_knowledge_index()
    
    def sync_from_database(self, conn: Any, cur: Any):
        """
        Sync vector indices from database (useful after initial setup or when loading existing data).
        
        Args:
            conn: Database connection
            cur: Database cursor
        """
        # Sync clinical notes
        cur.execute("SELECT id, embedding FROM clinical_notes")
        notes = cur.fetchall()
        
        for record_id, embedding_blob in notes:
            if record_id not in self.notes_labels:
                embedding = self._blob_to_embedding(embedding_blob)
                embedding_array = np.array([embedding], dtype=np.float32)
                label = self.next_notes_label
                self.notes_index.add_items(embedding_array, [label])
                self.notes_labels[record_id] = label
                self.next_notes_label += 1
        
        # Sync medical knowledge
        cur.execute("SELECT id, embedding FROM medical_knowledge")
        knowledge = cur.fetchall()
        
        for record_id, embedding_blob in knowledge:
            if record_id not in self.knowledge_labels:
                embedding = self._blob_to_embedding(embedding_blob)
                embedding_array = np.array([embedding], dtype=np.float32)
                label = self.next_knowledge_label
                self.knowledge_index.add_items(embedding_array, [label])
                self.knowledge_labels[record_id] = label
                self.next_knowledge_label += 1
        
        self.save_all_indices()
        print(f"Synced {len(notes)} notes and {len(knowledge)} knowledge entries")

def setup_database():
    """Setup the database using pyturso directly."""
    
    print("🔧 Setting up Turso Database for MediFlow V4")
    print(f"Database: {DB_PATH}")
    
    # Connect to database
    con = turso.connect(DB_PATH)
    cur = con.cursor()
    
    try:
        # 1. Drop existing tables (if any)
        print("   > Dropping existing tables...")
        cur.execute("DROP TABLE IF EXISTS clinical_notes")
        cur.execute("DROP TABLE IF EXISTS medical_knowledge")
        cur.execute("DROP TABLE IF EXISTS patients")
        
        # 2. Create patients table
        print("   > Creating patients table...")
        cur.execute("""
            CREATE TABLE patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                system_id TEXT UNIQUE NOT NULL,
                full_name TEXT NOT NULL,
                gov_id TEXT UNIQUE NOT NULL,
                phone TEXT,
                dob TEXT NOT NULL,
                gender TEXT
            )
        """)
        
        # 3. Create clinical_notes table with vector embedding and content hash
        print("   > Creating clinical_notes table...")
        cur.execute("""
            CREATE TABLE clinical_notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                visit_date TEXT NOT NULL,
                condition TEXT NOT NULL,
                note_content TEXT NOT NULL,
                embedding F32_BLOB(768),
                content_hash TEXT UNIQUE,
                FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE
            )
        """)
        
        # 4. Create medical_knowledge table with vector embedding and content hash
        print("   > Creating medical_knowledge table...")
        cur.execute("""
            CREATE TABLE medical_knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding F32_BLOB(768),
                content_hash TEXT UNIQUE,
                source_file TEXT
            )
        """)

        print("   > Creating ingestion_sources table...")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS ingestion_sources (
                filename TEXT PRIMARY KEY,
                file_hash TEXT NOT NULL,
                updated_at TEXT
            )
        """)

        # 5. Create indexes
        print("   > Creating indexes...")
        cur.execute("CREATE INDEX idx_patients_system_id ON patients(system_id)")
        cur.execute("CREATE INDEX idx_patients_full_name ON patients(full_name)")
        cur.execute("CREATE INDEX idx_clinical_notes_patient_id ON clinical_notes(patient_id)")
        cur.execute("CREATE INDEX idx_clinical_notes_condition ON clinical_notes(condition)")
        cur.execute("CREATE INDEX idx_clinical_notes_content_hash ON clinical_notes(content_hash)")
        cur.execute("CREATE INDEX idx_medical_knowledge_content_hash ON medical_knowledge(content_hash)")
        cur.execute("CREATE INDEX idx_medical_knowledge_topic ON medical_knowledge(topic)")
        cur.execute("CREATE INDEX idx_medical_knowledge_source_file ON medical_knowledge(source_file)")
        
        con.commit()
        
        # 6. Verify tables were created
        print("\n📊 Verifying tables:")
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = cur.fetchall()
        for table in tables:
            print(f"   ✓ {table[0]}")
        
        # 7. Test vector operations
        print("\n🧪 Testing vector operations...")
        
        # Create a test table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS vector_test (
                id INTEGER PRIMARY KEY,
                text TEXT,
                embedding F32_BLOB(768)
            )
        """)
        
        # Insert test vector
        test_vector = json.dumps([0.1] * 768)
        cur.execute("""
            INSERT INTO vector_test (text, embedding)
            VALUES (?, vector32(?))
        """, ("Test medical condition", test_vector))
        
        # Test vector distance calculation
        query_vector = json.dumps([0.2] * 768)
        cur.execute("""
            SELECT text, 
                   vector_distance_cos(embedding, vector32(?)) as distance
            FROM vector_test
            ORDER BY distance ASC
            LIMIT 1
        """, (query_vector,))
        
        result = cur.fetchone()
        if result:
            print(f"   ✓ Vector operations working!")
            print(f"     Text: {result[0]}, Distance: {result[1]:.4f}")
        
        # Clean up test table
        cur.execute("DROP TABLE IF EXISTS vector_test")
        
        con.commit()
        
        print("\n✅ Database setup complete!")
        
        # Initialize vector index
        print("\n🔍 Initializing vector index...")
        vector_index = MediFlowVectorIndex()
        print("✅ Vector index ready for use!")
        
        return vector_index
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise
    finally:
        cur.close()
        con.close()

if __name__ == "__main__":
    # When run directly, setup the database and vector index
    vector_index = setup_database()
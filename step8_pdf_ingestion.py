import os
import fitz  # PyMuPDF
import torch
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

DB_URI = "postgresql://aegis_admin:aegis_password@localhost:5432/aegis_med_db"
PDF_DIR = "./pdf_docs"

# Zero-Framework Chunker: Splits text by double-newlines and limits chunk size.
def chunk_text(text, max_words=150):
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = []
    current_length = 0

    for para in paragraphs:
        words = para.split()
        if current_length + len(words) > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.extend(words)
        current_length += len(words)
        
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def ingest_pdfs():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(" Loading BAAI/bge-m3 on H100 for PDF Embedding...")
    embedder = SentenceTransformer("BAAI/bge-m3", device=device)

    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith('.pdf')]
    if not pdf_files:
        print(" No PDFs found in ./pdf_docs/")
        return

    with psycopg.connect(DB_URI) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            for filename in pdf_files:
                print(f"\n Parsing: {filename}")
                filepath = os.path.join(PDF_DIR, filename)
                
                # 1. Read PDF with PyMuPDF
                doc = fitz.open(filepath)
                full_text = ""
                for page in doc:
                    full_text += page.get_text("text") + "\n\n"
                
                # 2. Chunk Text
                chunks = chunk_text(full_text)
                print(f"    Sliced into {len(chunks)} chunks. Embedding now...")
                
                # 3. Embed & Insert into PostgreSQL
                embeddings = embedder.encode(chunks, normalize_embeddings=True)
                
                for idx, (chunk_text_data, emb) in enumerate(zip(chunks, embeddings)):
                    if len(chunk_text_data.strip()) > 10:  # Skip empty chunks
                        cur.execute(
                            "INSERT INTO clinical_guidelines (source_title, drug_or_condition, content, embedding) VALUES (%s, %s, %s, %s)",
                            (filename, "Clinical Trial PDF", chunk_text_data, emb)
                        )
            conn.commit()
            print("\n All PDFs successfully embedded and inserted into Hybrid RAG DB!")

if __name__ == "__main__":
    ingest_pdfs()

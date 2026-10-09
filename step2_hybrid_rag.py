import torch
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer, CrossEncoder

DB_URI = "postgresql://aegis_admin:aegis_password@localhost:5432/aegis_med_db"

CLINICAL_CHUNKS = [
    ("FDA DailyMed: Levofloxacin Label (Normal Renal Function)", "Levofloxacin", "Community-Acquired Pneumonia: The usual dose of Levofloxacin is 500 mg orally or IV every 24 hours for 7 to 14 days, or 750 mg every 24 hours for 5 days in patients with normal renal function (CrCl >= 50 mL/min)."),
    ("FDA DailyMed: Levofloxacin Label (Renal Impairment Dosing)", "Levofloxacin", "Renal Impairment Dosage Adjustment: In patients with Community-Acquired Pneumonia and severe renal impairment (creatinine clearance CrCl 20 to 49 mL/min), administer a 750 mg loading dose, followed by 500 mg every 48 hours (q48h) to prevent drug accumulation and neurotoxicity."),
    ("StatPearls: Vancomycin Nephrotoxicity & CKD Protocol", "Vancomycin", "Vancomycin carries a significant risk of acute kidney injury (AKI), especially in patients with Stage 3b or Stage 4 CKD. Avoid concomitant nephrotoxins (such as ACE inhibitors like Lisinopril or NSAIDs) and dose strictly by trough or AUC/MIC monitoring."),
    ("StatPearls: ACE Inhibitors in Acute Kidney Injury", "Lisinopril", "Lisinopril and other ACE inhibitors dilate the efferent arteriole, reducing glomerular filtration pressure. In patients presenting with acute illness, sepsis, or rising serum creatinine (> 2.0 mg/dL) and hyperkalemia (potassium > 5.0 mEq/L), temporarily hold Lisinopril."),
    ("FDA DailyMed: Apixaban Renal & Cytochrome Interactions", "Apixaban", "For atrial fibrillation patients on Apixaban 5 mg twice daily, reduce dose to 2.5 mg twice daily only if at least two of the following are present: age >= 80 years, body weight <= 60 kg, or serum creatinine >= 1.5 mg/dL."),
    ("FDA DailyMed: Metformin Contraindications", "Metformin", "Metformin is contraindicated in patients with an eGFR below 30 mL/min/1.73m2 due to the risk of fatal metformin-associated lactic acidosis.")
]

def run_step2():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f" Using Compute Device: {device.upper()} ({torch.cuda.get_device_name(0)})")

    print(" Loading BAAI/bge-m3 and BAAI/bge-reranker-v2-m3...")
    embedder = SentenceTransformer("BAAI/bge-m3", device=device)
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device=device)

    with psycopg.connect(DB_URI) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE clinical_guidelines RESTART IDENTITY;")
            
            texts = [c[2] for c in CLINICAL_CHUNKS]
            embeddings = embedder.encode(texts, normalize_embeddings=True)

            for (source, drug, content), emb in zip(CLINICAL_CHUNKS, embeddings):
                cur.execute(
                    "INSERT INTO clinical_guidelines (source_title, drug_or_condition, content, embedding) VALUES (%s, %s, %s, %s)",
                    (source, drug, content, emb)
                )
            conn.commit()
            print(f" Embedded and stored {len(CLINICAL_CHUNKS)} clinical chunks in PostgreSQL.\n")

            query = "Levofloxacin renal dosing adjustment for CrCl 25 mL/min and holding Lisinopril in high creatinine"
            print(f" Clinical Query: '{query}'\n")
            query_vec = embedder.encode(query, normalize_embeddings=True)

            cur.execute(
                """
                WITH vector_search AS (
                    SELECT chunk_id, ROW_NUMBER() OVER (ORDER BY embedding <=> %s::vector) AS vec_rank
                    FROM clinical_guidelines LIMIT 20
                ),
                keyword_search AS (
                    SELECT chunk_id, ROW_NUMBER() OVER (ORDER BY ts_rank_cd(content_tsv, websearch_to_tsquery('english', %s)) DESC) AS kw_rank
                    FROM clinical_guidelines WHERE content_tsv @@ websearch_to_tsquery('english', %s) LIMIT 20
                )
                SELECT g.chunk_id, g.source_title, g.content, v.vec_rank, k.kw_rank,
                       COALESCE(1.0 / (60 + v.vec_rank), 0.0) + COALESCE(1.0 / (60 + k.kw_rank), 0.0) AS rrf_score
                FROM vector_search v
                FULL OUTER JOIN keyword_search k ON v.chunk_id = k.chunk_id
                JOIN clinical_guidelines g ON g.chunk_id = COALESCE(v.chunk_id, k.chunk_id)
                ORDER BY rrf_score DESC LIMIT 5;
                """,
                (query_vec, query, query)
            )
            candidates = cur.fetchall()

    pairs = [[query, row[2]] for row in candidates]
    ce_scores = reranker.predict(pairs)

    ranked_results = sorted(zip(candidates, ce_scores), key=lambda x: x[1], reverse=True)

    print(" FINAL HYBRID RAG + CROSS-ENCODER RE-RANKED RESULTS:")
    print("-" * 85)
    for (chunk_id, title, content, vec_rank, kw_rank, rrf_score), ce_score in ranked_results[:3]:
        print(f"[Chunk #{chunk_id}] {title}")
        print(f"    Vec Rank: {vec_rank} | Keyword Rank: {kw_rank} | RRF Score: {rrf_score:.5f} | Cross-Encoder Score: {ce_score:.4f}")
        print(f"    Content: {content}\n")

if __name__ == "__main__":
    run_step2()

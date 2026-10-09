CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE patients (
    patient_id VARCHAR(20) PRIMARY KEY,
    age INT NOT NULL,
    sex CHAR(1) NOT NULL,
    weight_kg NUMERIC(5,2) NOT NULL,
    chronic_conditions TEXT[],
    active_medications TEXT[]
);

CREATE TABLE patient_labs (
    lab_id SERIAL PRIMARY KEY,
    patient_id VARCHAR(20) REFERENCES patients(patient_id),
    test_name VARCHAR(50) NOT NULL,
    value NUMERIC(8,2) NOT NULL,
    unit VARCHAR(20) NOT NULL,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE clinical_guidelines (
    chunk_id SERIAL PRIMARY KEY,
    source_title VARCHAR(200) NOT NULL,
    drug_or_condition VARCHAR(100) NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1024),
    content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);

CREATE INDEX idx_guidelines_tsv ON clinical_guidelines USING GIN (content_tsv);

INSERT INTO patients (patient_id, age, sex, weight_kg, chronic_conditions, active_medications)
VALUES 
('PT-101', 68, 'M', 72.0, ARRAY['Stage 3b CKD', 'Atrial Fibrillation'], ARRAY['Apixaban 5mg', 'Lisinopril 10mg']),
('PT-102', 42, 'F', 64.5, ARRAY['Type 2 Diabetes'], ARRAY['Metformin 500mg']);

INSERT INTO patient_labs (patient_id, test_name, value, unit)
VALUES 
('PT-101', 'creatinine', 2.40, 'mg/dL'),
('PT-101', 'potassium', 5.10, 'mEq/L'),
('PT-102', 'creatinine', 0.90, 'mg/dL'),
('PT-102', 'glucose', 145.00, 'mg/dL');

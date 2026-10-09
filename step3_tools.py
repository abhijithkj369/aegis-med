import psycopg
import json

DB_URI = "postgresql://aegis_admin:aegis_password@localhost:5432/aegis_med_db"

# ---------------------------------------------------------
# Tool 1: Deterministic Clinical Math Calculator
# ---------------------------------------------------------
def calculate_crcl(age: int, weight_kg: float, creatinine: float, is_female: bool) -> str:
    """
    Calculates Creatinine Clearance (CrCl) using the Cockcroft-Gault equation.
    This dictates whether a patient needs renal dosage adjustments for drugs.
    """
    if creatinine <= 0:
        return "Error: Creatinine must be > 0."
    
    # Standard Cockcroft-Gault formula
    crcl = ((140 - age) * weight_kg) / (72 * creatinine)
    if is_female:
        crcl *= 0.85
        
    # Provide clinical interpretation
    category = "Normal"
    if crcl < 30:
        category = "Severe Renal Impairment (High Toxicity Risk)"
    elif crcl < 50:
        category = "Moderate Renal Impairment"
        
    return json.dumps({
        "calculated_crcl_ml_min": round(crcl, 1),
        "clinical_category": category
    })

# ---------------------------------------------------------
# Tool 2: Read-Only Database Tool (With Guardrail)
# ---------------------------------------------------------
def get_patient_profile_and_labs(patient_id: str) -> str:
    """
    Fetches the patient's demographics, chronic conditions, active meds, and recent labs.
    """
    # GUARDRAIL: Hardcoded parameterized SQL query prevents SQL Injection or DROP commands.
    query = """
        SELECT p.age, p.sex, p.weight_kg, p.chronic_conditions, p.active_medications,
               json_agg(json_build_object('test', l.test_name, 'value', l.value, 'unit', l.unit)) as labs
        FROM patients p
        LEFT JOIN patient_labs l ON p.patient_id = l.patient_id
        WHERE p.patient_id = %s
        GROUP BY p.age, p.sex, p.weight_kg, p.chronic_conditions, p.active_medications;
    """
    
    try:
        with psycopg.connect(DB_URI) as conn:
            with conn.cursor() as cur:
                cur.execute(query, (patient_id,))
                result = cur.fetchone()
                
                if not result:
                    return f"Error: Patient {patient_id} not found."
                
                profile = {
                    "patient_id": patient_id,
                    "age": result[0],
                    "sex": result[1],
                    "weight_kg": float(result[2]),
                    "conditions": result[3],
                    "medications": result[4],
                    "labs": result[5]
                }
                return json.dumps(profile, indent=2)
    except Exception as e:
        return f"Database Error: {str(e)}"

# ---------------------------------------------------------
# Test Execution
# ---------------------------------------------------------
def run_step3():
    print(" AEGIS-MED TOOL EXECUTION TEST\n" + "-"*40)
    
    # 1. Fetch PT-101's Labs
    print("1. Calling get_patient_profile_and_labs('PT-101')...")
    pt101_profile_json = get_patient_profile_and_labs("PT-101")
    print(pt101_profile_json)
    print("\n" + "-"*40)
    
    # 2. Extract variables for the calculator
    pt_data = json.loads(pt101_profile_json)
    creatinine_val = next(lab["value"] for lab in pt_data["labs"] if lab["test"] == "creatinine")
    
    # 3. Run the Clinical Calculator
    print("2. Calling calculate_crcl() with PT-101's data...")
    print(f"   Inputs: Age={pt_data['age']}, Weight={pt_data['weight_kg']}kg, Cr={creatinine_val}, Female={pt_data['sex'] == 'F'}")
    
    crcl_result = calculate_crcl(
        age=pt_data["age"],
        weight_kg=pt_data["weight_kg"],
        creatinine=float(creatinine_val),
        is_female=(pt_data["sex"] == 'F')
    )
    print(f"   Output: {crcl_result}")

if __name__ == "__main__":
    run_step3()

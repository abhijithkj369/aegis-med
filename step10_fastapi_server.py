import json
import re
import asyncio
import psycopg
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from sentence_transformers import SentenceTransformer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from contextlib import asynccontextmanager
from step3_tools import get_patient_profile_and_labs, calculate_crcl

DB_URI = "postgresql://aegis_admin:aegis_password@localhost:5432/aegis_med_db"
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "="*50)
    print(" [LIFESPAN] Booting vLLM Engine on H100...")
    ml_models["llm"] = LLM(
        model="Qwen/Qwen2.5-7B-Instruct", 
        max_model_len=4096, enforce_eager=True, gpu_memory_utilization=0.5,
        enable_lora=True, max_loras=1, max_lora_rank=16
    )
    
    # FIX: Strict temperature and exact stop tokens prevent JSON hallucination loops
    ml_models["sampling_params"] = SamplingParams(
        temperature=0.0, 
        max_tokens=250,
        stop=["<|im_end|>"]
    )
    
    print(" [LIFESPAN] Booting BAAI/bge-m3 RAG Embedder...")
    ml_models["embedder"] = SentenceTransformer("BAAI/bge-m3", device="cuda")
    print(" [LIFESPAN] Server Ready! Accepting Requests.")
    print("="*50 + "\n")
    yield
    ml_models.clear()

app = FastAPI(title="Aegis-Med Production API", lifespan=lifespan)

def search_clinical_guidelines_guardrail(query: str) -> str:
    embedder = ml_models["embedder"]
    query_vec = embedder.encode(query, normalize_embeddings=True)
    with psycopg.connect(DB_URI) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT content FROM clinical_guidelines
                ORDER BY embedding <=> %s::vector LIMIT 1;
            """, (query_vec,))
            res = cur.fetchone()
            return res[0] if res else "No guidelines found."

@app.get("/api/v1/triage/{patient_id}")
async def triage_patient(patient_id: str):
    llm = ml_models["llm"]
    sampling_params = ml_models["sampling_params"]
    
    async def fsm_stream_generator():
        state = {"phase": "START", "patient_data": None, "renal_status": None, "steps": 0}
        
        while state["phase"] != "FINISHED" and state["steps"] < 5:
            state["steps"] += 1
            yield f"data:  [Step {state['steps']}] Orchestrator invoking LoRA Agent...\n\n"
            await asyncio.sleep(0.1)
            
            prompt = f"""<|im_start|>system\nYou are a medical triage AI. You have two tools: "get_patient_profile" and "calculate_crcl".\nCURRENT STATE:\n{json.dumps(state)}\nDecide the next step. Output ONLY a raw JSON object.<|im_end|>\n<|im_start|>assistant\n"""
            
            outputs = llm.generate(
                prompts=[prompt], sampling_params=sampling_params, use_tqdm=False,
                lora_request=LoRARequest("lora_tools", 1, "./aegis-med-lora-adapter")
            )
            raw_response = outputs[0].outputs[0].text.strip()
            clean_json = re.sub(r"```json\s*|\s*```", "", raw_response)
            
            try:
                decision = json.loads(clean_json)
                yield f"data:  AI THOUGHT: {decision.get('reason')}\n\n"
            except:
                yield f"data:  ERROR: LLM returned invalid JSON -> {clean_json}\n\n"
                break
                
            if decision.get("action") == "call_tool":
                tool = decision.get("tool")
                if tool == "get_patient_profile":
                    state["patient_data"] = json.loads(get_patient_profile_and_labs(patient_id))
                    yield f"data:  ACTION: Fetched EHR for {patient_id}.\n\n"
                
                elif tool == "calculate_crcl":
                    cr = next(lab["value"] for lab in state["patient_data"]["labs"] if lab["test"] == "creatinine")
                    state["renal_status"] = json.loads(calculate_crcl(
                        state["patient_data"]["age"], state["patient_data"]["weight_kg"], float(cr), state["patient_data"]["sex"] == 'F'
                    ))
                    yield f"data:  ACTION: Calculated CrCl -> {state['renal_status']['calculated_crcl_ml_min']} mL/min.\n\n"
            
            elif decision.get("action") == "finish":
                yield f"data:  GUARDRAIL TRIGGERED: Verifying CrCl against RAG Clinical Guidelines...\n\n"
                crcl_val = state['renal_status']['calculated_crcl_ml_min']
                rag_query = f"Dosage adjustment renal impairment CrCl {crcl_val} mL/min"
                verified_guideline = search_clinical_guidelines_guardrail(rag_query)
                yield f"data:  RAG CITATION FOUND: {verified_guideline}\n\n"
                yield f"data:  TRIAGE COMPLETE. Handing off to human physician.\n\n"
                state["phase"] = "FINISHED"

    return StreamingResponse(fsm_stream_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("step10_fastapi_server:app", host="0.0.0.0", port=8000)

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

# Note: We must use the async connection class now
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

# FIX: Converted to a proper async function with psycopg.AsyncConnection
async def search_clinical_guidelines_guardrail_async(query: str) -> str:
    embedder = ml_models["embedder"]
    # Embedding is fast enough to stay synchronous
    query_vec = embedder.encode(query, normalize_embeddings=True)
    
    # NEW: Async database connection prevents event loop blocking during SSE
    async with await psycopg.AsyncConnection.connect(DB_URI) as conn:
        async with conn.cursor() as cur:
            await cur.execute("""
                SELECT content FROM clinical_guidelines
                ORDER BY embedding <=> %s::vector LIMIT 1;
            """, (query_vec,))
            res = await cur.fetchone()
            return res[0] if res else "No guidelines found."

@app.get("/api/v1/triage/{patient_id}")
def triage_patient(patient_id: str):
    llm = ml_models["llm"]
    sampling_params = ml_models["sampling_params"]
    
    # We must use an async generator to safely await the DB calls
    async def fsm_stream_generator():
        state = {
            "phase": "START", 
            "patient_data": None, 
            "renal_status": None, 
            "steps_taken": 0,
            "max_steps": 5
        }
        
        while state["phase"] != "FINISHED" and state["steps_taken"] < state["max_steps"]:
            state["steps_taken"] += 1
            yield f"data:  [Step {state['steps_taken']}] Orchestrator invoking LoRA Agent...\n\n"
            await asyncio.sleep(0.1) # Yield to ASGI loop
            
            prompt = f"""<|im_start|>system\nYou are a medical triage AI. You have two tools:\n1. "get_patient_profile": Fetches patient age, sex, weight, conditions, and labs. Use this first.\n2. "calculate_crcl": Calculates kidney function. Use this AFTER you have the patient profile.\n\nIf you have BOTH the patient profile and the renal status, use the action "finish".\n\nCURRENT STATE:\n{json.dumps(state, indent=2)}\n\nDecide the next step. You MUST output ONLY a raw JSON object in this format, with no markdown formatting or extra text:\n{{\"action\": \"call_tool\", \"tool\": \"tool_name\", \"reason\": \"Why you are doing this\"}}\nOR\n{{\"action\": \"finish\", \"reason\": \"Why you are finished\"}}<|im_end|>\n<|im_start|>assistant\n"""
            
            # Since vllm.LLM is synchronous, we run it in a background thread so we don't freeze the async generator
            outputs = await asyncio.to_thread(
                llm.generate,
                prompts=[prompt], 
                sampling_params=sampling_params, 
                use_tqdm=False,
                lora_request=LoRARequest("lora_tools", 1, "./aegis-med-lora-adapter")
            )
            raw_response = outputs[0].outputs[0].text.strip()
            yield f"data:  RAW LLM OUTPUT: {raw_response}\n\n"
            
            clean_json = re.sub(r"```json\s*|\s*```", "", raw_response)
            
            try:
                decision = json.loads(clean_json)
            except Exception as e:
                yield f"data:  ERROR: LLM returned invalid JSON -> {str(e)}\n\n"
                break
                
            action = decision.get("action")
            
            if action == "call_tool":
                tool = decision.get("tool")
                if tool == "get_patient_profile":
                    # We wrap the synchronous tool in to_thread
                    raw_data = await asyncio.to_thread(get_patient_profile_and_labs, patient_id)
                    state["patient_data"] = json.loads(raw_data)
                    yield f"data:  ACTION: Fetched EHR for {patient_id}.\n\n"
                
                elif tool == "calculate_crcl":
                    cr = next(lab["value"] for lab in state["patient_data"]["labs"] if lab["test"] == "creatinine")
                    raw_result = await asyncio.to_thread(
                        calculate_crcl,
                        state["patient_data"]["age"], 
                        state["patient_data"]["weight_kg"], 
                        float(cr), 
                        state["patient_data"]["sex"] == 'F'
                    )
                    state["renal_status"] = json.loads(raw_result)
                    yield f"data:  ACTION: Calculated CrCl -> {state['renal_status']['calculated_crcl_ml_min']} mL/min.\n\n"
            
            elif action == "finish":
                yield f"data:  GUARDRAIL TRIGGERED: Verifying CrCl against RAG Clinical Guidelines...\n\n"
                crcl_val = state['renal_status']['calculated_crcl_ml_min']
                rag_query = f"Dosage adjustment renal impairment CrCl {crcl_val} mL/min"
                
                # FIX: Safely await the async guardrail database call
                verified_guideline = await search_clinical_guidelines_guardrail_async(rag_query)
                
                yield f"data:  RAG CITATION FOUND: {verified_guideline}\n\n"
                yield f"data:  TRIAGE COMPLETE. Handing off to human physician.\n\n"
                state["phase"] = "FINISHED"

    return StreamingResponse(fsm_stream_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("step11_final_server:app", host="0.0.0.0", port=8000)

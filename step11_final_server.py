import json
import re
import asyncio
import psycopg
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
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
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

async def search_clinical_guidelines_guardrail_async(query: str) -> str:
    embedder = ml_models["embedder"]
    query_vec = embedder.encode(query, normalize_embeddings=True)
    async with await psycopg.AsyncConnection.connect(DB_URI) as conn:
        async with conn.cursor() as cur:
            vec_str = str(query_vec.tolist())
            await cur.execute("""
                SELECT content FROM clinical_guidelines
                ORDER BY embedding <=> %s::vector LIMIT 1;
            """, (vec_str,))
            res = await cur.fetchone()
            return res[0] if res else "No guidelines found."

@app.get("/api/v1/triage/{patient_id}")
def triage_patient(patient_id: str):
    llm = ml_models["llm"]
    sampling_params = ml_models["sampling_params"]
    
    async def fsm_stream_generator():
        state = {
            "phase": "START", "patient_data": None, "renal_status": None, "steps_taken": 0, "max_steps": 5, "agent_error": None
        }
        
        while state["phase"] != "FINISHED" and state["steps_taken"] < state["max_steps"]:
            state["steps_taken"] += 1
            yield f"data:  [Step {state['steps_taken']}] Orchestrator invoking LoRA Agent...\n\n"
            await asyncio.sleep(0.1) 
            
            # Note: We added the "agent_error" to the prompt so the AI knows when it messed up!
            prompt = f"""<|im_start|>system\nYou are a medical triage AI. You have two tools:\n1. "get_patient_profile": Fetches patient age, sex, weight, conditions, and labs. Use this first.\n2. "calculate_crcl": Calculates kidney function. Use this AFTER you have the patient profile.\n\nIf you have BOTH the patient profile and the renal status, use the action "finish".\n\nCURRENT STATE:\n{json.dumps(state, indent=2)}\n\nDecide the next step. You MUST output ONLY a raw JSON object in this format, with no markdown formatting or extra text:\n{{\"action\": \"call_tool\", \"tool\": \"tool_name\", \"reason\": \"Why you are doing this\"}}\nOR\n{{\"action\": \"finish\", \"reason\": \"Why you are finished\"}}<|im_end|>\n<|im_start|>assistant\n"""
            
            try:
                outputs = await asyncio.to_thread(
                    llm.generate, prompts=[prompt], sampling_params=sampling_params, use_tqdm=False,
                    lora_request=LoRARequest("lora_tools", 1, "./aegis-med-lora-adapter")
                )
                raw_response = outputs[0].outputs[0].text.strip()
                yield f"data:  RAW LLM OUTPUT: {raw_response}\n\n"
            except Exception as e:
                yield f"data:  MODEL ERROR: {str(e)}\n\n"
                break
            
            clean_json = re.sub(r"```json\s*|\s*```", "", raw_response)
            
            try:
                decision = json.loads(clean_json)
                state["agent_error"] = None # Clear previous errors
            except Exception as e:
                state["agent_error"] = f"Invalid JSON format: {str(e)}"
                yield f"data:  JSON ERROR: {str(e)}. Retrying...\n\n"
                continue
                
            action = decision.get("action")
            
            if action == "call_tool":
                tool = decision.get("tool")
                if tool == "get_patient_profile":
                    raw_data = await asyncio.to_thread(get_patient_profile_and_labs, patient_id)
                    state["patient_data"] = json.loads(raw_data)
                    yield f"data:  ACTION: Fetched EHR for {patient_id}.\n\n"
                
                elif tool == "calculate_crcl":
                    if not state["patient_data"]:
                        state["agent_error"] = "You must call get_patient_profile before calculating CrCl."
                        yield f"data:  AGENT BLOCKED: Tried to calculate math without data.\n\n"
                        continue
                        
                    cr = next(lab["value"] for lab in state["patient_data"]["labs"] if lab["test"] == "creatinine")
                    raw_result = await asyncio.to_thread(calculate_crcl, state["patient_data"]["age"], state["patient_data"]["weight_kg"], float(cr), state["patient_data"]["sex"] == 'F')
                    state["renal_status"] = json.loads(raw_result)
                    yield f"data:  ACTION: Calculated CrCl -> {state['renal_status']['calculated_crcl_ml_min']} mL/min.\n\n"
            
            elif action == "finish":
                # FIX 1: Hard block the AI if it tries to finish without doing the math!
                if not state["renal_status"]:
                    state["agent_error"] = "CRITICAL: You cannot finish without calling calculate_crcl first!"
                    yield f"data:  GUARDRAIL REJECTION: AI attempted to finish without computing CrCl. Forcing retry.\n\n"
                    continue
                
                yield f"data:  GUARDRAIL TRIGGERED: Verifying CrCl against RAG Clinical Guidelines...\n\n"
                crcl_val = state['renal_status']['calculated_crcl_ml_min']
                rag_query = f"Dosage adjustment renal impairment CrCl {crcl_val} mL/min"
                
                try:
                    verified_guideline = await search_clinical_guidelines_guardrail_async(rag_query)
                    yield f"data:  RAG CITATION FOUND: {verified_guideline}\n\n"
                    yield f"data:  TRIAGE COMPLETE. Handing off to human physician.\n\n"
                except Exception as e:
                    yield f"data:  DATABASE ERROR: {str(e)}\n\n"
                
                state["phase"] = "FINISHED"

    return StreamingResponse(fsm_stream_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("step11_final_server:app", host="0.0.0.0", port=8000)
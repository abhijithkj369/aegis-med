import json
import re
from vllm import LLM, SamplingParams
from step3_tools import get_patient_profile_and_labs, calculate_crcl

# 1. Load the 7B Model onto your H100 GPU
print(" Loading Qwen2.5-7B into H100 VRAM using vLLM (This takes ~60 seconds)...")
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct", 
    max_model_len=4096, 
    enforce_eager=True,  # Optimizes startup time for local scripts
    gpu_memory_utilization=0.5 # Only use 50% of the H100's 80GB VRAM
)
sampling_params = SamplingParams(temperature=0.0, max_tokens=150)
print(" Model loaded successfully!\n")

class RealMedicalAgent:
    def __init__(self, patient_id: str):
        self.patient_id = patient_id
        self.state = {
            "phase": "START",
            "patient_data": None,
            "renal_status": None,
            "steps_taken": 0,
            "max_steps": 5
        }

    def _llm_decide_next_action(self):
        """
        Builds a prompt injecting the current state and available tools,
        asking the LLM to output a JSON decision.
        """
        prompt = f"""<|im_start|>system
You are a medical triage AI. You have two tools:
1. "get_patient_profile": Fetches patient age, sex, weight, conditions, and labs. Use this first.
2. "calculate_crcl": Calculates kidney function. Use this AFTER you have the patient profile.

If you have BOTH the patient profile and the renal status, use the action "finish".

CURRENT STATE:
{json.dumps(self.state, indent=2)}

Decide the next step. You MUST output ONLY a raw JSON object in this format, with no markdown formatting or extra text:
{{"action": "call_tool", "tool": "tool_name", "reason": "Why you are doing this"}}
OR
{{"action": "finish", "reason": "Why you are finished"}}<|im_end|>
<|im_start|>assistant
"""
        # Generate the response using vLLM
        outputs = llm.generate(prompts=[prompt], sampling_params=sampling_params, use_tqdm=False)
        response_text = outputs[0].outputs[0].text.strip()
        
        # Clean up any accidental markdown blocks the LLM might add (e.g., ```json)
        response_text = re.sub(r"```json\s*", "", response_text)
        response_text = re.sub(r"\s*```", "", response_text)
        
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            return {"action": "error", "reason": f"LLM output invalid JSON: {response_text}"}

    def run_triage(self):
        print(f" Starting Real LLM Agent Loop for Patient: {self.patient_id}\n" + "="*50)
        
        while self.state["phase"] != "FINISHED":
            self.state["steps_taken"] += 1
            if self.state["steps_taken"] > self.state["max_steps"]:
                print(" CRITICAL: Max steps reached.")
                break

            #  The H100 Neural Network Thinks!
            print(f"[Step {self.state['steps_taken']}] LLM Thinking...")
            decision = self._llm_decide_next_action()
            
            if decision.get("action") == "error":
                print(f" {decision['reason']}")
                break
                
            print(f"    THOUGHT: {decision.get('reason')}")

            #  Execute Tools
            if decision.get("action") == "call_tool":
                if decision.get("tool") == "get_patient_profile":
                    raw_data = get_patient_profile_and_labs(self.patient_id)
                    self.state["patient_data"] = json.loads(raw_data)
                    print(f"    ACTION: get_patient_profile_and_labs() -> Success")
                
                elif decision.get("tool") == "calculate_crcl":
                    cr = next(lab["value"] for lab in self.state["patient_data"]["labs"] if lab["test"] == "creatinine")
                    raw_result = calculate_crcl(
                        age=self.state["patient_data"]["age"],
                        weight_kg=self.state["patient_data"]["weight_kg"],
                        creatinine=float(cr),
                        is_female=(self.state["patient_data"]["sex"] == 'F')
                    )
                    self.state["renal_status"] = json.loads(raw_result)
                    print(f"    ACTION: calculate_crcl() -> Success ({self.state['renal_status']['calculated_crcl_ml_min']} mL/min)")

            elif decision.get("action") == "finish":
                print(" FINISHED: LLM recognized all data is gathered.")
                self.state["phase"] = "FINISHED"

            print("-" * 50)
            
        return self.state

if __name__ == "__main__":
    agent = RealMedicalAgent("PT-101")
    agent.run_triage()

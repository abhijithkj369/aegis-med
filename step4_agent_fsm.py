import json

# 1. Import the tools we built in Step 3
from step3_tools import get_patient_profile_and_labs, calculate_crcl

class MedicalAgent:
    def __init__(self, patient_id: str):
        self.patient_id = patient_id
        # State Management: The Agent's Scratchpad Memory
        self.state = {
            "phase": "START",
            "patient_data": None,
            "renal_status": None,
            "final_diagnosis": None,
            "steps_taken": 0,
            "max_steps": 5  # Guardrail: Prevent infinite LLM loops
        }

    # 2. Simulated LLM Brain
    def _simulated_llm_decide_next_action(self):
        """
        In production, this is where we send the prompt to Llama-3 (vLLM).
        For now, we simulate the LLM's 'Thought' process deterministically.
        """
        if not self.state["patient_data"]:
            return {"action": "call_tool", "tool": "get_patient_profile", "reason": "I need the patient's EHR data."}
        
        if not self.state["renal_status"]:
            return {"action": "call_tool", "tool": "calculate_crcl", "reason": "I have the labs. I must calculate kidney function before dosing drugs."}
        
        return {"action": "finish", "reason": "I have all clinical data required."}

    # 3. The Pure-Python Finite State Machine Loop
    def run_triage(self):
        print(f" Starting Agent Loop for Patient: {self.patient_id}\n" + "="*50)
        
        while self.state["phase"] != "FINISHED":
            self.state["steps_taken"] += 1
            
            # Guardrail: Abort if looping too much
            if self.state["steps_taken"] > self.state["max_steps"]:
                print(" CRITICAL: Max steps reached. Aborting agent loop to prevent hallucination.")
                return self.state

            # Step A: The LLM "Thinks"
            decision = self._simulated_llm_decide_next_action()
            print(f"[Step {self.state['steps_taken']}]  THOUGHT: {decision['reason']}")

            # Step B: Execute the Tool based on LLM Decision
            if decision["action"] == "call_tool":
                if decision["tool"] == "get_patient_profile":
                    print("    ACTION: Executing SQL Tool -> get_patient_profile_and_labs()")
                    # Execute tool and save result to Agent State
                    raw_data = get_patient_profile_and_labs(self.patient_id)
                    self.state["patient_data"] = json.loads(raw_data)
                    print(f"    OBSERVATION: Fetched EHR for a {self.state['patient_data']['age']}yo {self.state['patient_data']['sex']}.")
                
                elif decision["tool"] == "calculate_crcl":
                    print("    ACTION: Executing Math Tool -> calculate_crcl()")
                    # Extract inputs from state
                    creatinine = next(lab["value"] for lab in self.state["patient_data"]["labs"] if lab["test"] == "creatinine")
                    
                    # Execute tool and save result to Agent State
                    raw_result = calculate_crcl(
                        age=self.state["patient_data"]["age"],
                        weight_kg=self.state["patient_data"]["weight_kg"],
                        creatinine=float(creatinine),
                        is_female=(self.state["patient_data"]["sex"] == 'F')
                    )
                    self.state["renal_status"] = json.loads(raw_result)
                    print(f"    OBSERVATION: Patient has {self.state['renal_status']['clinical_category']} (CrCl: {self.state['renal_status']['calculated_crcl_ml_min']} mL/min).")

            # Step C: Exit Condition
            elif decision["action"] == "finish":
                print(" FINISHED: Triage complete. Handing off to human physician.")
                self.state["phase"] = "FINISHED"

            print("-" * 50)
            
        return self.state

if __name__ == "__main__":
    agent = MedicalAgent("PT-101")
    final_state = agent.run_triage()

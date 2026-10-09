import torch
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR = "./aegis-med-lora-adapter"

# 1. Synthetic Golden Dataset
TRAINING_DATA = [
    {
        "prompt": """<|im_start|>system\nYou are a medical triage AI. You have two tools: "get_patient_profile" and "calculate_crcl".\nCURRENT STATE:\n{"phase": "START", "patient_data": null, "renal_status": null}\nDecide the next step. Output ONLY a raw JSON object.<|im_end|>\n<|im_start|>assistant\n""",
        "completion": """{"action": "call_tool", "tool": "get_patient_profile", "reason": "I need the patient's EHR data."}<|im_end|>"""
    },
    {
        "prompt": """<|im_start|>system\nYou are a medical triage AI. You have two tools: "get_patient_profile" and "calculate_crcl".\nCURRENT STATE:\n{"phase": "START", "patient_data": {"age": 68, "sex": "M", "weight_kg": 72.0}, "renal_status": null}\nDecide the next step. Output ONLY a raw JSON object.<|im_end|>\n<|im_start|>assistant\n""",
        "completion": """{"action": "call_tool", "tool": "calculate_crcl", "reason": "I have the patient profile, now I must calculate renal function."}<|im_end|>"""
    },
    {
        "prompt": """<|im_start|>system\nYou are a medical triage AI. You have two tools: "get_patient_profile" and "calculate_crcl".\nCURRENT STATE:\n{"phase": "START", "patient_data": {"age": 68, "sex": "M", "weight_kg": 72.0}, "renal_status": {"calculated_crcl_ml_min": 30.0}}\nDecide the next step. Output ONLY a raw JSON object.<|im_end|>\n<|im_start|>assistant\n""",
        "completion": """{"action": "finish", "reason": "I have all required clinical data."}<|im_end|>"""
    }
] * 50

def format_instruction(example):
    return {"text": example["prompt"] + example["completion"]}

def run_step6():
    print(" Preparing H100 GPU for LoRA Fine-Tuning...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    print(f" Loading {MODEL_ID} in 4-bit precision via BitsAndBytes...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # We load the base model...
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16
    )
    model = prepare_model_for_kbit_training(model)

    # Define the config, but DO NOT call get_peft_model()
    lora_config = LoraConfig(
        r=16, 
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"], 
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    dataset = Dataset.from_list(TRAINING_DATA)
    dataset = dataset.map(format_instruction)

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=5,
        max_steps=30,
        optim="paged_adamw_8bit",
        fp16=False,
        bf16=True,  
        save_strategy="no",
        report_to="none",
        dataset_text_field="text"
    )

    print("\n STARTING LORA TRAINING ON H100...")
    
    # We pass BOTH the base model and the lora_config to SFTTrainer.
    # SFTTrainer will wrap the model for us safely.
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        peft_config=lora_config
    )
    trainer.train()

    print(f"\n Saving LoRA adapter to {OUTPUT_DIR}...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    print(" Fine-Tuning Complete!")

if __name__ == "__main__":
    run_step6()

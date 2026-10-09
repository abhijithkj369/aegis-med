# Aegis-Med: Production Agentic AI & Hybrid RAG Engineering Log

## Hardware & Environment
- **GPU:** NVIDIA H100 NVL (80 GB HBM3) on Native Ubuntu.
- **Why Pivot to Native Linux:** Removes Windows kernel hypervisor overhead. The 80GB HBM3 memory allows Aegis-Med to deploy larger quantized models and handle massive clinical token contexts without OOM faults.
- **Runtime:** Python 3.12, Docker 28.4.0, CUDA 12.1.

---

## Step 1: Unified Database Architecture (PostgreSQL + pgvector + Redis)
- Deployed a containerized **PostgreSQL 16** instance and **Redis Stack** using Docker Compose.
- Designed a unified schema separating **Structured Patient Data** (`patients`, `patient_labs`) from **Unstructured Clinical Knowledge** (`clinical_guidelines`).
- Verified database queries and Redis state persistence.

---

## Step 2: GPU Hybrid RAG Pipeline (RRF + Cross-Encoder Re-Ranking)

### 1. What Was Built
- Executed `step2_hybrid_rag.py` locally on the **NVIDIA H100 GPU**.
- Used **`BAAI/bge-m3`** (1024-dim dense bi-encoder) to embed clinical guidelines into PostgreSQL `pgvector`.
- Implemented a single-query **Hybrid Search** in PostgreSQL combining Dense Semantic Search and Sparse Keyword Search via **Reciprocal Rank Fusion (RRF)**.
- Added a second-stage **Cross-Encoder Re-Ranker (`BAAI/bge-reranker-v2-m3`)** on CUDA to score `[query, chunk]` pairs jointly.

### 2. Core Engineering Concepts Learned
- **Cross-Encoder Efficacy in Medicine:** The pipeline successfully differentiated between standard Levofloxacin dosing (Cross-Encoder score: `0.4861`) and the required renal impairment dosing (Cross-Encoder score: `0.6855`) based on the specific query context of "CrCl 25 mL/min".

---

## Step 3: Deterministic Clinical Tools & SQL Guardrails
### 1. What Was Built
- Created `step3_tools.py` containing a **Tool Registry** for the Agent.
- **SQL Guardrail Tool:** Built `get_patient_profile_and_labs()` using parameterized SQL (`%s`) to strictly fetch EHR data, blocking any possibility of LLM SQL-injection or destructive `DROP/DELETE` commands.
- **Deterministic Math Tool:** Built `calculate_crcl()` to execute the Cockcroft-Gault equation in pure Python. LLMs are probabilistic text generators and should *never* perform clinical math themselves.

---

## Step 4: Zero-Framework Agent Orchestration (FSM)
### 1. What Was Built
- Created `step4_agent_fsm.py` to orchestrate tools without LangChain or LangGraph.
- Built a **Finite State Machine (FSM)** based on the ReAct (Reason + Act) loop: `Thought -> Action -> Observation`.
- Implemented a **Retry Budget Guardrail** (`max_steps = 5`) to forcibly terminate the loop if the agent gets confused, preventing infinite compute loops in production.

### 2. Interview Talking Point
> "I avoid high-level frameworks like LangChain for Agent orchestration. Instead, I build deterministic Finite State Machines in pure Python. This guarantees I have strict Human-in-the-Loop intervention points, exact retry budgets, and total control over the state dictionary, which is critical for clinical safety."

---

## Step 5: Qwen2.5-7B Agent via vLLM
### 1. What Was Built
- Loaded `Qwen/Qwen2.5-7B-Instruct` locally onto the H100 GPU using **vLLM**, optimizing for throughput and latency.
- Discovered the "Infinite Loop Problem" where off-the-shelf models drift into conversational prose instead of strictly formatting JSON tool calls, causing the FSM parser to fail and trigger the `max_steps` abort sequence.

---

## Step 6: QLoRA Fine-Tuning for Strict JSON Execution
### 1. What Was Built
- Generated a synthetic "Golden Dataset" mapping clinical FSM state JSONs to strictly formatted tool-call JSONs.
- Executed a Low-Rank Adaptation (LoRA) fine-tuning job via Hugging Face `trl` (`SFTTrainer`) on the H100.
- Target Modules: `q_proj`, `v_proj` | Rank: `r=16`.
- The training completed 30 epochs in 25 seconds, dropping loss from `2.76` to `0.02` while only updating `0.06%` of the model parameters.

---

## Step 7: Dynamic LoRA Hot-Swapping in Production
### 1. What Was Built
- Rewrote the Agent Orchestrator to utilize vLLM's **Dynamic LoRA Hot-Swapping** via `LoRARequest("clinical_tools_lora", 1, "./aegis-med-lora-adapter")`.
- The Base 7B model remains permanently resident in GPU VRAM, while the tiny 5MB adapter is injected into the active computation stream strictly at inference time.
- The Agent successfully completed the end-to-end clinical loop (fetching the profile, calculating CrCl, and terminating) using perfectly formatted JSON schema.

### 2. Interview Talking Point
> "I don't rely on prompt engineering alone to force LLMs to output JSON. In Aegis-Med, I generated a golden dataset and QLoRA fine-tuned an open-source 7B model specifically for tool execution. I served the base model via vLLM on an H100 and hot-swapped the LoRA adapter dynamically per-request. This guarantees 100% schema adherence without the latency of calling an external frontier API."

---

## Step 8: Expanding Hybrid RAG with Real PDFs
### 1. What Was Built
- Integrated `PyMuPDF` to parse unstructured clinical trial PDFs directly into the PostgreSQL RAG database.
- Implemented a zero-framework Python chunking algorithm that splits documents semantically (by paragraph) to ensure clinical context is not severed mid-sentence.
- Embedded chunks dynamically using the H100's Tensor Cores (`BAAI/bge-m3`) and inserted them into the `pgvector` index.

---

## Steps 9–11: FastAPI Streaming, ASGI Architecture & Guardrails
### 1. What Was Built
- Wrapped the FSM Orchestrator in a **FastAPI** web service using **Server-Sent Events (SSE)** via `StreamingResponse`.
- Built a **Verification Guardrail**: The API intercepts the AI's final "finish" decision and forces a synchronous RAG lookup against the clinical guidelines to verify the computed CrCl value before streaming the final response to the user.
- **Resolved ASGI Blocking Conflicts:** Discovered that synchronous GPU calls (`vLLM.generate`) and synchronous DB drivers (`psycopg`) block the ASGI event loop, causing curl connection drops (`curl: (18) transfer closed`). 
- **Production Architecture Fix:** Refactored the architecture to initialize the massive ML models inside a FastAPI `@asynccontextmanager` Lifespan event, used `asyncio.to_thread` to push heavy GPU computation to a background worker pool, and utilized `psycopg.AsyncConnection` for non-blocking database queries during the SSE stream.

### 2. Interview Talking Point
> "A major failure mode in production AI is blocking the ASGI event loop during streaming responses. In Aegis-Med, I decoupled the vLLM inference engine and the PostgreSQL RAG lookups from the main FastAPI thread. I initialize the 7B model and embedding engines strictly within the app lifespan, and use `asyncio.to_thread` with async DB drivers to stream Server-Sent Events without dropping HTTP connections. Furthermore, I built a final-layer verification guardrail that intercepts the Agent's conclusion and strictly cites a retrieved RAG document before the response is allowed to reach the end-user."

# Project Conclusion
Aegis-Med successfully demonstrates a zero-framework, fully local, open-source Agentic AI system capable of autonomous tool execution, Hybrid RAG, LoRA hot-swapping, and production-grade safety guardrails.

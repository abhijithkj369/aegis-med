# Aegis-Med: Autonomous Clinical Triage and Hybrid RAG Architecture

Aegis-Med is a production-grade Agentic AI backend designed for clinical triage applications. Developed from first principles without relying on high-level abstraction frameworks (such as LangChain or LangGraph), the system orchestrates a localized 7B-parameter Large Language Model, deterministic clinical tools, and a PostgreSQL-backed Hybrid RAG pipeline entirely on an NVIDIA H100 GPU.

## Core Architecture

* **Zero-Framework Orchestration:** A pure-Python Finite State Machine (FSM) enforcing the ReAct loop (Thought, Action, Observation) with strict retry budgets and fail-safes to prevent infinite generation cycles.
* **Dynamic LoRA Hot-Swapping:** Leverages the vLLM engine to keep the base Qwen2.5-7B model permanently resident in GPU VRAM, while dynamically injecting a custom QLoRA adapter—fine-tuned specifically for strict JSON schema adherence—at inference time.
* **Unified Hybrid RAG:** Utilizes PostgreSQL 16 equipped with `pgvector` and `tsvector` to execute Reciprocal Rank Fusion (RRF) and Cross-Encoder re-ranking, eliminating the need for fragmented vector databases.
* **Asynchronous Streaming (SSE):** A robust FastAPI backend utilizing `asyncio.to_thread` and `psycopg.AsyncConnection` to stream Server-Sent Events to the client interface without blocking the ASGI event loop or dropping database connections.
* **Deterministic Clinical Guardrails:** A final safety layer that intercepts the FSM's concluding state, triggering an automated RAG semantic search to verify calculated metrics (e.g., Creatinine Clearance) against established medical guidelines before returning the final diagnosis.

## Technology Stack

* **Compute & Serving:** NVIDIA H100 (Native Ubuntu), vLLM (PagedAttention architecture), PyTorch
* **Models:** Qwen/Qwen2.5-7B-Instruct (Base), Custom QLoRA Adapter (r=16, 5M parameters), BAAI/bge-m3, BAAI/bge-reranker-v2-m3
* **Model Fine-Tuning:** Hugging Face `trl` (SFTTrainer), `peft`, `BitsAndBytes` (nf4 quantization)
* **Database & Document Ingestion:** PostgreSQL 16, pgvector, Redis Stack, PyMuPDF
* **API Layer:** FastAPI, Server-Sent Events (SSE), Uvicorn

## Execution Workflow

1. **Patient Context Retrieval:** The FSM executes a parameterized, read-only SQL tool to securely fetch patient EHR data, strictly mitigating SQL-injection vulnerabilities.
2. **Clinical Mathematics:** The LLM triggers a pure-Python Cockcroft-Gault calculator to compute Creatinine Clearance (CrCl), isolating deterministic mathematics from probabilistic language generation.
3. **RAG Verification:** Upon reaching a preliminary diagnostic state, the asynchronous event loop triggers a semantic search against embedded clinical trial PDFs to cite exact dosing guidelines based on the computed CrCl metric.

---

**Author**
Abhijith K J | AI Engineer, Centre for Development of Advanced Computing (CDAC)

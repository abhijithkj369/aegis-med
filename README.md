#  Aegis-Med: Production Agentic AI & Hybrid RAG

Aegis-Med is a production-grade, zero-framework Agentic AI backend for clinical triage. Built from first principles without LangChain or LangGraph, this system orchestrates a local 7B-parameter LLM, deterministic clinical tools, and a PostgreSQL-backed Hybrid RAG pipeline directly on an NVIDIA H100 GPU.

##  Core Architecture

* **Zero-Framework Orchestration:** A pure-Python Finite State Machine (FSM) enforcing the ReAct loop (Thought → Action → Observation) with strict retry budgets and human-in-the-loop exit states.
* **Dynamic LoRA Hot-Swapping:** Utilizes `vLLM` to keep a Qwen2.5-7B base model resident in GPU VRAM, while dynamically injecting a custom QLoRA adapter (trained on a synthetic JSON tool-use dataset) strictly at inference time.
* **Unified Hybrid RAG:** Replaces fragmented vector databases with PostgreSQL 16 (`pgvector` + `tsvector`), executing Reciprocal Rank Fusion (RRF) and Cross-Encoder re-ranking for clinical guideline retrieval.
* **Asynchronous Streaming (SSE):** A FastAPI backend utilizing `asyncio.to_thread` and `psycopg.AsyncConnection` to stream AI agent thoughts live to the frontend without blocking the ASGI event loop or dropping database connections.
* **Clinical Guardrails:** Enforces a deterministic safety layer that intercepts the FSM's final state, triggering an automated RAG citation lookup to verify calculated metrics (e.g., Creatinine Clearance) against established medical guidelines before concluding triage.

##  Tech Stack

* **Compute & Serving:** NVIDIA H100 (Native Ubuntu), `vLLM` (PagedAttention), PyTorch.
* **Models:** `Qwen/Qwen2.5-7B-Instruct` (Base), Custom QLoRA Adapter (r=16, 5M params), `BAAI/bge-m3`, `BAAI/bge-reranker-v2-m3`.
* **Fine-Tuning:** Hugging Face `trl` (`SFTTrainer`), `peft`, `BitsAndBytes` (nf4 quantization).
* **Database & Ingestion:** PostgreSQL 16, `pgvector`, Redis Stack, PyMuPDF.
* **API Layer:** FastAPI, Server-Sent Events (SSE), Uvicorn.

##  Execution Flow

1. **Patient Context Fetch:** The FSM executes a parameterized, read-only SQL tool to fetch patient EHR data, preventing SQL-injection hallucinations.
2. **Deterministic Math:** The LLM triggers a pure-Python Cockcroft-Gault calculator for Creatinine Clearance (CrCl), isolating probabilistic generation from clinical mathematics.
3. **RAG Verification:** Upon reaching a diagnosis, the asynchronous event loop triggers a semantic `pgvector` search against embedded clinical trial PDFs to cite exact dosing adjustments based on the computed CrCl.

##  Author
**Abhijith K J** | AI Engineer

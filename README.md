# Aegis-Med: Agentic Clinical Decision Support with Hybrid RAG

Aegis-Med is an AI engineering project exploring agentic workflows for clinical decision support and pharmacovigilance. It combines a deterministic Python orchestrator, a clinical language model, structured EHR data access, deterministic clinical calculators, and hybrid retrieval-augmented generation (RAG).

The system is designed to make clinical AI workflows more structured, traceable, and easier to evaluate, with explicit tool boundaries and human oversight.

> **Disclaimer:** Aegis-Med is a research and engineering project, not a substitute for professional medical judgment. It is not intended for autonomous diagnosis, treatment, or prescribing. Outputs must be independently reviewed by qualified healthcare professionals.

## Key Features

- **Deterministic orchestration:** A Python finite-state machine (FSM) manages agent execution, tool calls, state transitions, and bounded retries.
- **LLM-powered tool selection:** A locally served language model generates structured tool requests that are validated before execution.
- **Structured EHR access:** Read-only SQL tools restrict database access to approved operations and schemas.
- **Hybrid RAG:** Combines dense vector retrieval and PostgreSQL full-text search, with Reciprocal Rank Fusion (RRF) and reranking where implemented.
- **Clinical calculators:** Deterministic Python functions perform supported clinical calculations independently of the language model.
- **Evidence-aware responses:** Retrieved clinical references can be attached to results to support review and traceability.
- **Streaming API:** A FastAPI backend can stream sanitized progress events using Server-Sent Events (SSE).
- **Safety controls:** Input validation, tool authorization, bounded execution, and output checks help reduce unsafe or unintended behavior.

Feature descriptions should reflect the components implemented and tested in the current repository.

## System Architecture

```mermaid
flowchart TD
    Client[Client Application] --> API[FastAPI API]
    API --> Auth[Authentication and Request Validation]
    Auth --> FSM[Python FSM Orchestrator]

    FSM --> LLM[LLM Inference Server]
    LLM --> Parser[Structured Output Validation]
    Parser --> Policy[Tool Authorization and Policy Checks]
    Policy --> Tools[Approved Tool Registry]

    Tools --> EHR[Read-only PostgreSQL EHR Queries]
    Tools --> Calc[Deterministic Clinical Calculators]
    Tools --> RAG[Hybrid Retrieval Pipeline]

    RAG --> Dense[pgvector Dense Search]
    RAG --> Lexical[PostgreSQL Full-text Search]
    Dense --> Fusion[RRF and Optional Reranking]
    Lexical --> Fusion
    Fusion --> Evidence[Retrieved Evidence and Metadata]

    EHR --> FSM
    Calc --> FSM
    Evidence --> FSM

    FSM --> Review[Output Validation and Human Review]
    Review --> Stream[Sanitized SSE Events]
    Stream --> Client


The diagram describes the logical architecture. Mark components as implemented only after verifying them against the codebase.
Technology Stack
Layer	Technologies
Language	Python
Orchestration	Custom finite-state machine
Model serving	vLLM
Language model	Configured base model and adapter
Fine-tuning	QLoRA, PEFT, Hugging Face TRL, where used
Structured data	PostgreSQL
Vector retrieval	pgvector
Lexical retrieval	PostgreSQL full-text search (tsvector)
Retrieval ranking	RRF and cross-encoder, where implemented
API	FastAPI, Uvicorn, SSE
Cache or auxiliary services	Redis Stack, if used
Deployment	Docker Compose, where configured
Request Lifecycle
1. Receive request: Validate the incoming request and check access permissions.
2. Initialize orchestration: The FSM creates a bounded execution context.
3. Request model output: The language model generates a structured response or proposed tool call.
4. Validate the action: Validate the output schema, tool name, arguments, and authorization rules.
5. Execute the tool: An approved tool retrieves permitted data, performs a deterministic calculation, or searches clinical references.
6. Update state: The FSM records the tool result and decides whether another step is needed.
7. Validate the response: Check required output fields, supporting evidence, and applicable safety rules.
8. Return the result: Stream approved status events and return a result for human review.
Model-generated actions must never be treated as executable code or trusted SQL.
Hybrid RAG
The retrieval pipeline combines complementary search methods:
- Dense retrieval: Searches clinical document embeddings using pgvector.
- Lexical retrieval: Searches terms and phrases using PostgreSQL full-text search.
- Rank fusion: Combines candidate rankings using Reciprocal Rank Fusion.
- Reranking: An optional cross-encoder reranks retrieved candidates.
- Evidence handling: Retains source identifiers and relevant metadata for citations.
The final ranking pipeline should reflect the actual implementation. Retrieval relevance does not establish that a clinical statement is correct, complete, or applicable to a particular patient.
Clinical Tools and Data Access
Clinical calculations are implemented as deterministic functions rather than delegated to free-form language generation.
Document each calculator with:
- Intended clinical use and required inputs.
- Units, accepted ranges, and missing-data behavior.
- Formula source and version.
- Tests covering normal, boundary, and invalid inputs.
- Known limitations and situations requiring clinician review.
For example, Cockcroft-Gault estimates creatinine clearance (CrCl). It should not be described as interchangeable with estimated glomerular filtration rate (eGFR).
Database tools should use parameterized queries, read-only database permissions, explicit table and column allowlists, bounded query execution, and application-level authorization. SQL validation alone is not a security boundary.
API Documentation
Document the actual routes from the FastAPI application and its OpenAPI schema.
Example Request
The following command is illustrative. Replace the route, identifier, and authentication details with those implemented in the repository.
curl -N \
  -H "Authorization: Bearer $AEGIS_API_TOKEN" \
  -H "Accept: text/event-stream" \
  "http://localhost:8000/<actual-triage-route>"


Example Sanitized Event Stream
event: status
data: {"stage":"retrieval","status":"started"}

event: tool_result
data: {"tool":"clinical_reference_search","status":"completed"}

event: evidence
data: {"source_id":"guideline-example","status":"retrieved"}

event: complete
data: {"status":"completed","requires_clinical_review":true}


These events illustrate a possible API contract, not verified output from the running application. Do not expose hidden model reasoning, raw prompts, access tokens, unredacted patient records, or unrestricted tool results to clients.
Document request parameters, authentication, response schemas, event types, error handling, timeouts, and cancellation behavior.
Installation and Configuration
The commands below are a general setup outline. Exact commands depend on the repository's dependency files and deployment configuration.
Prerequisites
- Python version supported by the project dependencies.
- Access to the configured language model and adapter.
- PostgreSQL with the required extensions, including pgvector if used.
- NVIDIA GPU and compatible drivers if running GPU inference.
- Docker and Docker Compose if using the containerized deployment.
Setup
git clone <your-aegis-med-repository-url>
cd <repository-directory>

python -m venv .venv


Activate the environment on Linux:
source .venv/bin/activate


Install dependencies using the repository's actual dependency manifest. For example:
pip install -r requirements.txt


Configure required environment variables using a local environment file based on a committed .env.example. Do not commit real credentials.
Before starting the application, initialize the database, apply the required migrations, verify model availability, and confirm that API and database health checks pass.
Configuration
Create a documented .env.example containing placeholders for settings actually used by the application.
AEGIS_ENV=development
AEGIS_API_HOST=127.0.0.1
AEGIS_API_PORT=8000

DATABASE_URL=postgresql://<user>:<password>@<host>:5432/<database>

MODEL_ID=<configured-model-id>
LORA_ADAPTER_PATH=<configured-adapter-path>

LOG_LEVEL=INFO


This is a configuration template, not a guarantee that these exact environment variable names are supported. Map them to the application's settings implementation before use.
Do not expose the database or inference server publicly without appropriate access controls. Keep secrets out of Git, logs, screenshots, and example outputs.
Evaluation and Benchmarks
Performance and quality should be reported using reproducible experiments rather than estimated figures.
Metric	Measurement to Report
Model startup time	Seconds, hardware, and loading configuration
Peak GPU memory	MiB or GiB, precision, and workload
LoRA adapter size	Actual file size and trainable parameter count
Fine-tuning performance	Dataset, configuration, runtime, and evaluation results
Retrieval latency	Median and p95 latency, corpus size, and index configuration
API latency	Time to first event and total response time
Clinical calculation correctness	Unit tests and reference cases
Retrieval quality	Recall@k, MRR, or another appropriate metric
Answer quality	Faithfulness, relevance, citation correctness, and expert review
Add benchmark values only after running the corresponding experiments and documenting their methodology.
Security and Privacy
Clinical information requires particular care. Before deployment with real patient data:
- Enforce authentication, role-based authorization, and least-privilege access.
- Use read-only database credentials for retrieval tools.
- Validate model-generated tool requests and enforce argument schemas.
- Protect against prompt injection in user inputs and retrieved documents.
- Apply appropriate PII handling and data-retention controls.
- Restrict access to model and adapter administration endpoints.
- Avoid logging sensitive clinical data or internal prompts unnecessarily.
- Record auditable tool executions and relevant security events.
- Test timeouts, retry limits, cancellation, malformed outputs, and failure paths.
- Require human review for clinical decisions.
These controls must be tested and documented before claiming compliance with a healthcare privacy or security standard.
Testing
For a pytest-based project, run:
pytest -q


Recommended test categories include:
- FSM transitions, retry limits, and termination behavior.
- Tool schema validation and authorization.
- SQL safety and database permissions.
- Clinical calculation unit tests.
- Hybrid retrieval ranking and citation integrity.
- SSE streaming, disconnects, and error handling.
- Prompt-injection and malformed-model-output cases.
- End-to-end integration tests using synthetic data.
Document which tests exist, which pass, and which require external services or GPU hardware.
Limitations
- Model outputs can be incorrect or incomplete.
- Retrieval may miss relevant evidence or return outdated sources.
- Deterministic calculations are only as reliable as their inputs, formulas, and validation.
- Clinical guidance varies by context and may change over time.
- GPU memory, concurrency, and latency depend on hardware and serving configuration.
- Clinical evaluation is required before any use beyond research and development.
Roadmap
Potential future improvements include:
- Reproducible evaluation datasets and benchmark reports.
- Stronger end-to-end security and adversarial testing.
- Improved observability and structured audit trails.
- Versioned clinical references and evidence freshness checks.
- Expanded integration and failure-recovery tests.
- Clinician-led evaluation of usability and output quality.
Contributing
Contributions are welcome through issues and pull requests. Please include relevant tests, document configuration changes, and never submit real patient data, credentials, or other sensitive information.
License
Specify the actual project license here.
If the project uses the MIT License, include the full license text in a root-level LICENSE file before advertising it as MIT-licensed.
Author
Abhijith K J
AI Engineer | Medical AI, LLMs, and Agentic Systems
Centre for Development of Advanced Computing (CDAC)

# 🧠 Interview Backend – Technical README

This backend implements a **dual-interface AI interview system** consisting of:

* An **HTTP-based interview API** (FastAPI)
* A **real-time voice interview agent** (LiveKit Agents)

Both interfaces are powered by **LangGraph** for deterministic conversation flow and shared interview state modeling. The system supports CV-based personalization, session persistence, and multi-turn interview orchestration.

---

## 📘 Table of Contents

1. Backend Components
2. LangGraph Interview Engine
3. FastAPI Backend

   * Interview Lifecycle Endpoints
   * LiveKit Token & Room Provisioning
4. LiveKit Voice Agent Backend

   * Agent Entrypoint
   * Active Graph vs Full Graph
5. State Management & Checkpointing
6. Environment Configuration
7. Running the Backend Services

---

## 1. Backend Components

The backend is composed of the following primary modules:

* **FastAPI API server** (`interview.py`)
* **LiveKit voice agent worker** (`livekit_agent2.py`)
* **LangGraph interview graphs** (shared conceptual model)
* **Vector store services** for CV-based retrieval
* **LLM integrations** (Gemini, Tavily)

Each component is isolated by responsibility but interoperates through structured state and metadata.

---

## 2. LangGraph Interview Engine

Interview logic is implemented using **LangGraph**, where the interview is represented as a directed state graph. Each node corresponds to a specific interview stage, such as setup, question generation, answer evaluation, retrieval, or final feedback.

### State Model

Both the HTTP and voice systems rely on a structured interview state containing fields such as:

* `topic`, `question_type`
* `step`, `max_steps`
* `questions`, `answers`, `feedback`
* `cv_content`, `retrieved_context`
* `needs_retrieval`, `similarity_score`
* `waiting_for_user`, `final_evaluation`

State is passed between nodes and updated deterministically on each graph invocation.

---

## 3. FastAPI Backend (`interview.py`)

The FastAPI backend exposes endpoints for **text-based interviews** and **LiveKit session provisioning**.

### 3.1 Start Interview

**Endpoint**: `POST /start_interview`

**Responsibilities**:

* Create a new interview thread (`thread_id`)
* Accept job title and question type
* Optionally process uploaded CV PDFs
* Extract and chunk CV text
* Initialize vector stores for retrieval
* Initialize LangGraph interview state
* Invoke the interview graph to generate the first question

The endpoint returns the first interview question along with step metadata.

---

### 3.2 Continue Interview

**Endpoint**: `POST /continue_interview`

**Responsibilities**:

* Resume an existing interview session using `thread_id`
* Retrieve checkpointed LangGraph state
* Inject the user’s response into state
* Resume graph execution
* Route output as either:

  * Next interview question
  * Final evaluation and feedback

The endpoint supports multi-turn interviews and cleans up vector stores upon completion.

---

### 3.3 Join Voice Interview (LiveKit)

**Endpoint**: `POST /join`

**Responsibilities**:

* Accept username, job title, room name, question type, and optional CV
* Process CV text for context
* Initialize interview state for voice sessions
* Embed `initial_state` into LiveKit room metadata
* Generate a LiveKit access token
* Dispatch the `voice-agent` to the room

This endpoint bridges the HTTP backend with the real-time voice agent.

---

## 4. LiveKit Voice Agent Backend (`livekit_agent2.py`)

The LiveKit backend implements a **real-time AI interviewer** using LiveKit Agents and LangGraph.

### 4.1 Agent Entrypoint

The agent is registered using:

* `@server.rtc_session(agent_name="voice-agent")`

On session start, the agent:

* Connects to the LiveKit room
* Reads `initial_state` from room metadata
* Initializes a LangGraph workflow
* Speaks the first interview question

---

### 4.2 Active LangGraph Workflow

The currently active workflow is a **latency-optimized LangGraph** consisting of:

* **Setup Node**: Generates the first interview question
* **Echo Node**: Sends user responses to Gemini and returns generated replies
* **Router Logic**: Advances steps only when valid user input is detected

Checkpointing is handled using **MemorySaver**, scoped to the LiveKit session thread.

This minimal graph is designed for:

* Real-time speech interaction
* Low latency response generation
* Predictable conversational flow

---

### 4.3 Full Interview Graph

A full-featured interview graph is present and is responsible for

* Setup and initialization
* Answer collection
* Retrieval decision logic
* Vector-based CV retrieval (RAG)
* Tavily web search fallback
* Question generation
* Answer evaluation
* Final interview assessment
* Result display

It also supports persistent checkpointing with:

* PostgreSQL (preferred)
* SQLite fallback
* In-memory fallback


---

## 5. State Management & Checkpointing

* **FastAPI interviews** use LangGraph checkpoints keyed by `thread_id`
* **Voice interviews** use session-scoped thread IDs
* Checkpointing enables resumable interviews and deterministic replay

Persistence strategies vary by interface and latency requirements.

---

## 6. Environment Configuration

Required environment variables include:

* `LIVEKIT_API_KEY`
* `LIVEKIT_API_SECRET`
* `LIVEKIT_URL`
* `ELEVENLABS_API_KEY`
* Gemini API credentials
* Tavily API key

All configuration values should be defined in a `.env` file.

---

## 7. Running the Backend Services

### FastAPI Server

```bash
uvicorn main:app --reload
```

### LiveKit Agent Worker

```bash
python livekit_agent2.py
```

The LiveKit agent will activate automatically when dispatched to a room via the `/join` endpoint.

---

## ✅ Summary

This backend provides a **shared interview engine** exposed through both HTTP and real-time voice interfaces. By combining FastAPI, LiveKit Agents, LangGraph, retrieval services, and LLMs, the system supports structured, resumable, and context-aware technical interviews across multiple interact

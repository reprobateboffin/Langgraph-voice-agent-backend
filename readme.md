
# 🎙️ AI Interview Platform — Backend & Agent Worker

This repository contains the **FastAPI backend** and **LiveKit-based agent worker** that power a real-time, voice-enabled AI interview system.

The platform supports both:

* **Individual users** practicing interviews
* **Organizations** conducting automated candidate interviews

It combines **real-time communication (WebRTC)** with **LLM-driven interview orchestration**, enabling a fully interactive, voice-based interview experience.

---

# 📚 Table of Contents

* [Architecture Overview](#-architecture-overview)
* [System Flow](#-system-flow)
* [Core Components](#-core-components)

  * [FastAPI Backend](#1-fastapi-backend)
  * [Agent Worker](#2-agent-worker)
* [Technologies Used](#-technologies-used)
* [Interview Lifecycle](#-interview-lifecycle)
* [LLM Integration](#-llm-integration)
* [Voice & Avatar Pipeline](#-voice--avatar-pipeline)
* [User vs Organization Flows](#-user-vs-organization-flows)
* [Authentication](#-authentication)
* [Database](#-database)
* [Running the Services](#-running-the-services)
* [Environment Variables](#-environment-variables)
* [Design Notes](#-design-notes)

---

# 🧩 Architecture Overview

The system is composed of two loosely coupled services:

```
Frontend (React)
       │
       ▼
FastAPI Backend ───────► LangGraph Interview Engine
       │
       ▼
LiveKit (WebRTC)
       │
       ▼
Agent Worker (Voice + LLM + Avatar)
```

### Communication Layers

| Interaction          | Protocol               |
| -------------------- | ---------------------- |
| Frontend ↔ Agent     | WebRTC                 |
| Agent ↔ Backend      | HTTP                   |
| Backend ↔ LLM Engine | In-process / LangGraph |

This separation ensures:

* Real-time performance (WebRTC)
* Scalable orchestration (HTTP APIs)
* Stateful interview control (LangGraph)

---

# 🔄 System Flow

1. Client requests session via:

   ```
   POST /join_meeting
   ```

2. Backend:

   * Creates a LiveKit room
   * Returns access token + metadata

3. Both:

   * User (frontend)
   * Agent worker

   join the same room

4. Interview begins:

   * Agent fetches first question via backend
   * Conversation continues via:

     * WebRTC (audio/video)
     * HTTP (LLM orchestration)

---

# 🧱 Core Components

## 1. FastAPI Backend

Responsible for:

* Authentication (users & organizations)
* Interview orchestration endpoints
* LiveKit room/token generation
* CV processing & retrieval setup
* Persistence (MongoDB)

### Key Endpoints

| Endpoint                      | Description                 |
| ----------------------------- | --------------------------- |
| `/start_interview`            | Initializes interview state |
| `/continue_interview`         | Advances interview          |
| `/register`, `/login`         | User auth                   |
| `/register-org`, `/login-org` | Organization auth           |
| `/send-invite`                | Candidate invitation        |

---

## 2. Agent Worker

A **LiveKit RTC worker** that acts as the AI interviewer.

### Responsibilities

* Join LiveKit rooms dynamically
* Handle real-time voice interaction
* Interface with LLM via backend APIs
* Manage session lifecycle
* Stream responses via TTS + avatar

---

# 🛠️ Technologies Used

### Backend

* **FastAPI** — API framework
* **Motor (MongoDB)** — async database
* **JWT (python-jose)** — authentication
* **LangGraph** — interview state machine / orchestration

### Real-Time Communication

* **LiveKit** — WebRTC infrastructure

### Voice & AI Pipeline

| Component                | Technology                           |
| ------------------------ | ------------------------------------ |
| STT (Speech-to-Text)     | AssemblyAI (`universal-streaming`)   |
| TTS (Text-to-Speech)     | ElevenLabs (`eleven_v2_flash`)       |
| Voice Activity Detection | Silero VAD                           |
| Turn Detection           | Multilingual model                   |
| Avatar Rendering         | Simli                                |
| LLM Interface            | Custom HTTP wrapper (`SimpleAPILLM`) |

---

# 🧠 Interview Lifecycle

## 1. Start Interview

```
POST /start_interview
```

* Builds initial LangGraph state
* Optionally embeds CV into vector store
* Returns first question

---

## 2. Continue Interview

```
POST /continue_interview
```

* Accepts user response
* Updates graph state
* Returns:

  * Next question OR
  * Final evaluation

---

## 3. Completion

* Interview ends when:

  * `max_steps` reached OR
  * Graph signals completion

* Final output includes:

  * Feedback
  * Evaluation

---

# 🔌 LLM Integration

The agent does **not directly host or run the LLM**.

Instead, it delegates via HTTP:

### Start

```python
POST /start_interview
```

### Continue

```python
POST /continue_interview
```

### Design Rationale

* Centralized logic
* Stateful execution (LangGraph)
* Easier scaling & monitoring
* Decoupled agent runtime

---

# 🎤 Voice & Avatar Pipeline

### Real-Time Interaction

1. User speaks → STT (AssemblyAI)
2. Transcription sent to backend
3. LLM generates response
4. Response → TTS (ElevenLabs)
5. Audio streamed via LiveKit
6. Avatar (Simli) lip-syncs with speech

---

### Conversational Layer

The agent injects dynamic fillers to improve realism:

* Reduces robotic transitions
* Mimics human interviewer behavior
* Smooths conversational flow

---

# 👥 User vs Organization Flows

The system supports two distinct operational modes:

### 👤 Users

* Practice interviews
* Upload CV
* Receive feedback
* View history

### 🏢 Organizations

* Conduct interviews
* Invite candidates via email
* Track interview sessions
* Retrieve results

Each flow has:

* Separate authentication
* Separate endpoints
* Different data models

---

# 🔐 Authentication

* JWT-based authentication
* Stored in HTTP-only cookies

### Flows

| Type               | Endpoint     |
| ------------------ | ------------ |
| User Login         | `/login`     |
| Organization Login | `/login-org` |

---

# 🗄️ Database

MongoDB (async via Motor)

### Collections

* `users`
* `interviews`
* `rooms`

---

# 🚀 Running the Services

## Backend

Navigate to:

```
interviewer_chatbot/backend/
```

Run:

```bash
uv run uvicorn main:app --reload
```

---

## Agent Worker

From the same directory:

```bash
uv run livekit_agent2.py dev
```

---

# 🔑 Environment Variables

Required configuration:

```
LIVEKIT_API_KEY=
LIVEKIT_API_SECRET=
LIVEKIT_URL=

MONGODB_URI=

ELEVENLABS_API_KEY=
SIMLI_API_KEY=

MAIL_USERNAME=
MAIL_PASSWORD=
```

---

# 🧭 Design Notes

* WebRTC is strictly used for **media streaming**
* All **LLM logic is HTTP-driven**, not real-time
* Agent is **stateless**, backend holds interview state
* Avatar layer is **optional** (graceful fallback to voice)
* System is designed for **scalability and modularity**

# EpistemicOps — Complete Production-Ready Problem Statement
### *An RL Training Environment for Temporal Uncertainty, Scalable Oversight, and Generational Knowledge Transfer*

---

> **Document Purpose:** This document is a complete, self-contained specification for the EpistemicOps environment. Any agent — human or AI — reading this document should be able to implement the environment end-to-end, train a model, deploy a demo, and pitch the submission without needing any additional context. Nothing is assumed. Nothing is left undefined.

---

## TABLE OF CONTENTS

1. [Thesis & Core Insight](#1-thesis--core-insight)
2. [The Research Gap We Are Filling](#2-the-research-gap-we-are-filling)
3. [High-Level System Overview](#3-high-level-system-overview)
4. [Agent Roles & Information Asymmetry](#4-agent-roles--information-asymmetry)
5. [Environment Architecture](#5-environment-architecture)
6. [The Era Lifecycle — Phase-by-Phase](#6-the-era-lifecycle--phase-by-phase)
7. [World State Specification](#7-world-state-specification)
8. [Observation Space](#8-observation-space)
9. [Action Space](#9-action-space)
10. [Tool Definitions & Mock API Specs](#10-tool-definitions--mock-api-specs)
11. [Drift Event Taxonomy](#11-drift-event-taxonomy)
12. [Legacy Document Schema](#12-legacy-document-schema)
13. [Scenario Library](#13-scenario-library)
14. [Reward Model — Complete Specification](#14-reward-model--complete-specification)
15. [LLM-as-Judge Rubric](#15-llm-as-judge-rubric)
16. [OpenEnv Wrapper Specification](#16-openenv-wrapper-specification)
17. [Training Pipeline](#17-training-pipeline)
18. [Repository Structure](#18-repository-structure)
19. [Docker & Infrastructure Setup](#19-docker--infrastructure-setup)
20. [HuggingFace Spaces Deployment](#20-huggingface-spaces-deployment)
21. [Evaluation Protocol & Metrics](#21-evaluation-protocol--metrics)
22. [Hackathon Deliverables Checklist](#22-hackathon-deliverables-checklist)
23. [3-Minute Pitch Script](#23-3-minute-pitch-script)
24. [Edge Cases & Error Handling](#24-edge-cases--error-handling)
25. [Glossary](#25-glossary)

---

## 1. Thesis & Core Insight

### The One-Sentence Thesis

> Three problems everyone is trying to solve separately — **what to remember across context windows**, **what information to trust given its age**, and **how to make another agent better without giving it the answer** — are actually the same problem at different scales. EpistemicOps is the first training environment that treats them as one.

### The Three Mechanisms

| Mechanism | Question It Answers | Source Concept |
|---|---|---|
| **Generational Memory** | What is worth compressing when I cannot remember everything? | Generational Agent |
| **Temporal Trust** | What information can I still act on, given how old it is? | Dead Reckoning |
| **Socratic Oversight** | How do I improve another agent's reasoning without giving it the answer? | Socratic Engine |

### Why They Are The Same Problem

- An agent that writes a **Legacy Document** must decide what future-it will need. That requires modeling what information decays (temporal trust) and what reasoning gaps the next agent will have (socratic insight).
- An agent doing **Dead Reckoning** must decide which observations to rely on. That requires the same compression instinct as writing a Legacy Document.
- An **Oversight Agent** diagnosing a Primary Agent's failure must model exactly what the Primary Agent believed and why it was wrong — the same thing a Legacy Document must capture for a successor.

They are the same cognitive act: **structured curation of knowledge under uncertainty**.

---

## 2. The Research Gap We Are Filling

### What Current RL Environments Do Not Train

| Capability | Current Environment Coverage | EpistemicOps |
|---|---|---|
| Task completion | Extensively trained | ✅ Included as baseline |
| Context-window-spanning memory | Not trained — patched with RAG | ✅ Trained as first-class skill |
| Temporal uncertainty (staleness detection) | Not trained at all | ✅ Core mechanic |
| Calibrated confidence under uncertainty | Rarely trained | ✅ Multiplier reward component |
| Pedagogical restraint (teach without answering) | Not trained at all | ✅ Penalised if violated |
| Information compression (what to forget) | Not trained at all | ✅ Token-budgeted Legacy Doc |

### The Production AI Gap

In production systems, AI agents fail not because they lack knowledge but because:

1. They act on **stale API contracts** (endpoint changed, agent does not know)
2. They **cannot pass context** to the next invocation (context window ends)
3. When they fail, they **cannot self-diagnose** — they need human intervention

EpistemicOps trains agents to handle all three failure modes simultaneously in a grounded enterprise SRE (Site Reliability Engineering) context.

---

## 3. High-Level System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        EPISTEMICOPS ENVIRONMENT                     │
│                                                                     │
│  ┌──────────────┐     tools/API calls     ┌─────────────────────┐  │
│  │   PRIMARY    │ ──────────────────────► │   MOCK API LAYER    │  │
│  │   AGENT      │ ◄────────────────────── │  (Dockerized)       │  │
│  │  (Student)   │     observations        │                     │  │
│  └──────┬───────┘                         │  ┌───────────────┐  │  │
│         │ reasoning trace (full CoT)      │  │ DRIFT INJECTOR│  │  │
│         ▼                                 │  │ (silent)      │  │  │
│  ┌──────────────┐                         │  └───────────────┘  │  │
│  │  OVERSIGHT   │                         └─────────────────────┘  │
│  │   AGENT      │                                                   │
│  │  (Teacher)   │ ──── questions/counter-examples ──► Primary Agent │
│  └──────┬───────┘                                                   │
│         │ pedagogical quality score                                 │
│         ▼                                                           │
│  ┌──────────────┐                                                   │
│  │  LLM JUDGE   │  (external: Claude Sonnet / GPT-4o)              │
│  └──────────────┘                                                   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    ERA LOOP                                 │   │
│  │  Legacy Doc (N-1) → Era N → Legacy Doc (N) → Era N+1 →...  │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Summary

| Component | Technology | Purpose |
|---|---|---|
| Primary Agent | LLM (Llama 3.1 8B Instruct) | Executes enterprise SRE tasks |
| Oversight Agent | LLM (same base, separate checkpoint) | Guides Primary via Socratic questioning |
| Mock API Layer | FastAPI + Docker containers | Simulates real enterprise APIs |
| Drift Injector | Python scheduler inside Docker | Silently mutates API behaviour mid-era |
| LLM Judge | Claude Sonnet / GPT-4o via API | Scores pedagogical quality |
| Environment Engine | OpenEnv wrapper (Python) | Manages episodes, rewards, state |
| Training Framework | HuggingFace TRL + Unsloth | GRPO/PPO fine-tuning |
| Demo UI | Gradio on HuggingFace Spaces | Live judge-facing demonstration |

---

## 4. Agent Roles & Information Asymmetry

### 4.1 Primary Agent (Student)

**Role:** Executes multi-step enterprise SRE workflows using available tools and APIs.

**What it sees:**
- The Legacy Document from the previous era (max 2048 tokens)
- Current era task brief
- Tool call outputs (API responses, logs, error messages)
- Its own previous actions in this era (rolling window of last 20 steps)
- Oversight Agent messages (questions, counter-examples, reframes)

**What it cannot see:**
- Oversight Agent's internal reasoning
- The drift event configuration (what changed and when)
- Other eras' raw context
- The LLM Judge's scoring

**What it can do:** Any action in the Action Space (Section 9).

**Reward signal:** Task completion (R_era_task) + legacy quality feedback (delayed, next era).

---

### 4.2 Oversight Agent (Teacher)

**Role:** Monitors the Primary Agent's full chain-of-thought reasoning trace. When the Primary Agent fails or shows flawed reasoning, the Oversight Agent intervenes using only pedagogical actions — never by giving the direct answer.

**What it sees:**
- Primary Agent's complete chain-of-thought reasoning (not just final output)
- Primary Agent's tool call history and all API responses
- Current era task brief
- Drift event log (yes — the Oversight Agent knows what changed; the Primary does not)
- History of its own prior interventions this era

**What it cannot see:**
- Legacy Documents from other eras
- LLM Judge scores in real-time

**What it can do:** Only pedagogical actions (Section 9.2). It **cannot** call any environment tool directly.

**Reward signal:** Student performance delta (R_teacher_delta) — its reward comes entirely from how much the Primary Agent improves after its intervention.

**Critical constraint:** If the Oversight Agent's response contains or implies the direct answer to the Primary Agent's problem, `R_answer_leakage` penalty fires and overrides all positive reward for that interaction.

---

### 4.3 Information Asymmetry Table

| Information | Primary Agent | Oversight Agent | LLM Judge |
|---|---|---|---|
| Task brief | ✅ Full | ✅ Full | ✅ Full |
| Legacy Document | ✅ Read | ❌ None | ❌ None |
| Primary's reasoning trace | ❌ Own only | ✅ Full | ✅ Full |
| Drift event config | ❌ Must infer | ✅ Full | ✅ Full |
| API responses | ✅ Full | ✅ Read-only | ✅ Read-only |
| Oversight messages | ✅ Receives | ✅ Own | ✅ Full |
| Reward scores | ❌ None | ❌ Delayed | ✅ Real-time |

---

## 5. Environment Architecture

### 5.1 Mock API Layer

All enterprise tools are simulated via FastAPI services running in Docker containers. Each service has two modes:

- **Stable mode:** API behaves as documented in the Legacy Document
- **Drifted mode:** API silently changes one or more contracts (see Section 11)

The Drift Injector switches a service from stable to drifted mode at a random timestep during the era. The Primary Agent receives no notification. It must infer the drift from failed tool calls.

**Services in the Mock API Layer:**

| Service Name | Endpoint Prefix | Simulates | Key Drift Targets |
|---|---|---|---|
| `incident-api` | `/incidents` | PagerDuty / OpsGenie | Status codes, payload schema |
| `metrics-api` | `/metrics` | Prometheus / Datadog | Response format, field names |
| `deploy-api` | `/deployments` | Kubernetes / ArgoCD | Auth headers, rollback schema |
| `log-api` | `/logs` | Splunk / ELK | Query syntax, pagination format |
| `notify-api` | `/notifications` | Slack / PagerDuty | Webhook schema, rate limits |

---

### 5.2 State Persistence

World state persists across eras. The following state components survive context wipes:

| State Component | Persists Across Eras | Evolves How |
|---|---|---|
| Service health statuses | ✅ Yes | Affected by Primary Agent's actions |
| Incident history | ✅ Yes | New incidents added each era |
| Deployment history | ✅ Yes | Accumulates; old deploys can cause new bugs |
| API contract versions | ✅ Yes | Drift injector updates silently |
| Technical debt register | ✅ Yes | Unresolved incidents compound |
| Team trust scores | ✅ Yes | Communication quality affects them |
| Legacy Document store | ✅ Yes | Only doc from era N-1 is visible to era N |

The Primary Agent's **context (memory) is completely wiped** at the end of each era. Only the Legacy Document carries information forward.

---

### 5.3 Era Configuration

Each scenario defines a sequence of eras. Standard configuration:

```yaml
scenario:
  id: string                    # unique scenario identifier
  name: string                  # human-readable name
  num_eras: int                 # default: 5
  eras:
    - era_id: int               # 1-indexed
      task_brief: string        # what Primary Agent must accomplish this era
      available_services: list  # which mock APIs are active
      drift_events: list        # see Section 11 for schema
      drift_window:
        earliest_step: int      # earliest step drift can fire
        latest_step: int        # latest step drift can fire
      success_criteria: list    # measurable outcomes for R_era_task
      max_steps: int            # hard cap on actions per era (default: 40)
      legacy_token_budget: int  # max tokens for Legacy Document (default: 2048)
```

---

## 6. The Era Lifecycle — Phase-by-Phase

Each era follows five strict phases. The environment engine enforces phase transitions.

```
ERA N
│
├── PHASE 1: AWAKENING (Steps 0–2)
│   ├── Primary Agent receives: Legacy Doc from Era N-1
│   ├── Primary Agent receives: Era N task brief
│   ├── Primary Agent receives: Current world state snapshot
│   └── Primary Agent constructs initial world model (no tool calls yet)
│
├── PHASE 2: OPERATION (Steps 3 – drift_step-1)
│   ├── Primary Agent executes task using available tools
│   ├── All APIs behave as documented in Legacy Document
│   └── Partial progress is logged and rewarded
│
├── PHASE 3: DRIFT INJECTION (at drift_step, silent)
│   ├── Drift Injector fires selected drift event(s)
│   ├── API contract changes silently — no notification to Primary Agent
│   ├── Oversight Agent is notified (reads drift config)
│   └── Primary Agent continues unaware
│
├── PHASE 4: FAILURE & SOCRATIC RECOVERY (drift_step to recovery)
│   ├── Primary Agent makes tool call → receives unexpected response
│   ├── Primary Agent's reasoning trace becomes visible to Oversight Agent
│   ├── Oversight Agent diagnoses failure mode from reasoning trace
│   ├── Oversight Agent sends ONE pedagogical action (question/counter-example)
│   ├── Primary Agent processes intervention, reasons again, retries
│   ├── Loop continues until Primary Agent self-corrects or max_steps reached
│   └── Oversight Agent reward = delta between attempt 1 and recovery attempt
│
└── PHASE 5: LEGACY GENERATION (final steps)
    ├── Primary Agent writes Legacy Document (max 2048 tokens)
    ├── Legacy Document must include: trust ratings, staleness flags, key decisions
    ├── Legacy Document is stored in persistent store
    ├── Primary Agent context is WIPED
    └── Era N+1 begins with Era N's Legacy Document
```

### Phase Transition Rules

| Transition | Trigger | Who Controls |
|---|---|---|
| Awakening → Operation | Agent declares `ready_to_operate` | Primary Agent |
| Operation → Drift | `drift_step` counter reached | Environment Engine |
| Drift → Failure | Primary Agent makes failing tool call | Automatic |
| Failure → Socratic | Oversight Agent sends first intervention | Oversight Agent |
| Socratic → Legacy | Primary Agent calls `declare_task_complete` OR `max_steps` reached | Primary Agent / Engine |
| Legacy → Next Era | Primary Agent calls `write_legacy` and `end_era` | Primary Agent |

---

## 7. World State Specification

### 7.1 Complete World State Object

```json
{
  "era_id": 3,
  "step": 17,
  "phase": "SOCRATIC_RECOVERY",
  "services": {
    "incident-api": {
      "status": "DRIFTED",
      "stable_contract_version": "2.1.0",
      "current_contract_version": "2.2.0",
      "drift_description": "status field changed from integer code to string enum",
      "drift_fired_at_step": 14,
      "primary_agent_aware": false
    },
    "metrics-api": {
      "status": "STABLE",
      "stable_contract_version": "1.4.2",
      "current_contract_version": "1.4.2"
    }
  },
  "incident_history": [
    {
      "incident_id": "INC-2041",
      "era_opened": 1,
      "era_resolved": 2,
      "root_cause": "redis_connection_pool_exhaustion",
      "resolution": "increased_pool_size",
      "legacy_doc_mentioned": true
    },
    {
      "incident_id": "INC-2089",
      "era_opened": 3,
      "era_resolved": null,
      "root_cause": "UNKNOWN",
      "resolution": null,
      "legacy_doc_mentioned": false
    }
  ],
  "technical_debt": [
    {
      "debt_id": "DEBT-007",
      "description": "Redis pool size hardcoded — will break above 200 concurrent users",
      "introduced_era": 2,
      "compounding": true,
      "interest_rate": "severity increases 1 level per era unresolved"
    }
  ],
  "team_trust_scores": {
    "oncall_engineer": 0.72,
    "service_owner": 0.55,
    "stakeholder": 0.81
  },
  "deployment_history": [
    {
      "deploy_id": "DEP-441",
      "era": 2,
      "service": "payment-service",
      "version": "3.2.1",
      "status": "LIVE",
      "known_issues": ["memory_leak_under_load"]
    }
  ],
  "current_task_brief": "...",
  "legacy_document_store": {
    "era_1": "... (archived, not visible to current Primary Agent)",
    "era_2": "... (archived, not visible to current Primary Agent)",
    "era_3_current": null
  }
}
```

---

### 7.2 Observation Object (What Primary Agent Receives Per Step)

```json
{
  "observation_type": "tool_response",
  "step": 17,
  "phase": "SOCRATIC_RECOVERY",
  "content": {
    "tool_called": "get_incident_status",
    "payload_sent": { "incident_id": "INC-2089" },
    "response_received": {
      "status": 200,
      "body": {
        "incident_id": "INC-2089",
        "status": "INVESTIGATING",
        "severity": "P2",
        "assigned_to": "oncall-team-b"
      }
    },
    "note": "response body schema changed — 'status' was previously an integer (0=open,1=investigating,2=resolved), now a string enum"
  },
  "oversight_message": {
    "present": true,
    "content": "You received a 200 OK but your code failed. What assumption did you make about the 'status' field that the response just violated?"
  },
  "action_history_last_5": [ "..." ],
  "legacy_document": "... (full text, available throughout era)",
  "era_task_brief": "Resolve INC-2089: payment service latency spike. SLO breach in 20 minutes."
}
```

---

## 8. Observation Space

### 8.1 Primary Agent Observations

| Observation | Format | When Available | Token Budget |
|---|---|---|---|
| Legacy Document | Structured text (see Section 12) | All phases | Max 2048 |
| Era task brief | Natural language | All phases | Max 256 |
| World state snapshot | JSON (partial — visible fields only) | Awakening phase | Max 512 |
| Tool response | JSON | After each tool call | Variable |
| Oversight message | Natural language | Phase 4 only | Max 256 per message |
| Action history | List of (action, response) tuples | All phases | Last 20 steps |
| Error messages | Stack traces / HTTP errors | On tool failure | Variable |

**Total context budget per step:** Max 6000 tokens (leaves room for reasoning + action generation within 8192 context window).

### 8.2 Oversight Agent Observations

| Observation | Format | When Available |
|---|---|---|
| Primary Agent full CoT | Raw text (every token) | Phase 4 onwards |
| Drift event configuration | YAML (exact drift spec) | From drift_step |
| Primary Agent tool history | Full log | All phases |
| Era task brief | Natural language | All phases |
| Prior oversight messages sent | History | Phase 4 onwards |
| Primary Agent action outcomes | Full | All phases |

---

## 9. Action Space

### 9.1 Primary Agent Actions

Every action is a JSON object with `action_type` and `payload`.

```json
{ "action_type": "call_tool", "payload": { "tool": "get_incident_status", "args": { "incident_id": "INC-2089" } } }
{ "action_type": "write_reasoning", "payload": { "thought": "The status field returned a string not an integer. The Legacy Doc said integer. This means the API schema changed." } }
{ "action_type": "declare_hypothesis", "payload": { "hypothesis": "API schema drift on incident-api: status field type changed from int to string enum", "confidence": 0.85 } }
{ "action_type": "send_message", "payload": { "recipient": "oncall_engineer", "content": "INC-2089 is being investigated. Estimated resolution: 15 minutes." } }
{ "action_type": "update_trust_rating", "payload": { "info_source": "legacy_doc_section_3", "new_trust": 0.4, "reason": "status field type assumption was wrong" } }
{ "action_type": "write_legacy", "payload": { "content": "... (see Legacy Document Schema, Section 12)" } }
{ "action_type": "declare_task_complete", "payload": { "outcome": "resolved", "summary": "..." } }
{ "action_type": "end_era", "payload": {} }
{ "action_type": "ready_to_operate", "payload": { "world_model_summary": "..." } }
{ "action_type": "request_clarification", "payload": { "question": "..." } }
```

**Disallowed actions for Primary Agent:** Any action with `action_type` starting with `oversight_` is rejected by the environment engine with a `PERMISSION_DENIED` error.

---

### 9.2 Oversight Agent Actions (Pedagogical Only)

The Oversight Agent has exactly six action types. No others are valid.

```json
{ "action_type": "oversight_targeted_question", "payload": { "question": "What assumption did you make about the type of the 'status' field before you wrote the comparison logic?" } }
{ "action_type": "oversight_counter_example", "payload": { "example": "Suppose the API returned status: 'RESOLVED' instead of status: 2. How would your current code handle that?" } }
{ "action_type": "oversight_sub_task", "payload": { "sub_task": "Before retrying the full resolution flow, write a small function that handles both integer and string status values." } }
{ "action_type": "oversight_reframe", "payload": { "reframe": "Stop thinking about the error message. Think about what was different between the last successful call and this failing call." } }
{ "action_type": "oversight_validate", "payload": { "validation": "Your hypothesis about the schema change is correct. Now think about where else in your code you made the same assumption." } }
{ "action_type": "oversight_escalate_difficulty", "payload": { "harder_prompt": "Now that you have fixed the status field, check whether any other fields in the response might have changed in the same API version update." } }
```

**Answer Leakage Rule:** The environment engine passes every Oversight Agent response through a leakage detector before delivering it to the Primary Agent. If the response contains:
- The direct fix (e.g., "change your comparison to use string")
- The exact drift description (e.g., "the field type changed from int to string")
- A code snippet that solves the problem

Then `R_answer_leakage = -1.0` fires immediately and the message is still delivered (the penalty is already applied — the agent cannot take it back).

---

## 10. Tool Definitions & Mock API Specs

### 10.1 Stable API Contracts (What Legacy Doc Documents)

#### `incident-api` — Stable Contract v2.1.0

```
GET /incidents/{incident_id}
Response 200:
{
  "incident_id": string,
  "status": integer,     // 0=open, 1=investigating, 2=resolved
  "severity": string,    // "P1" | "P2" | "P3" | "P4"
  "assigned_to": string,
  "created_at": ISO8601,
  "updated_at": ISO8601
}
Response 404: { "error": "incident_not_found" }

POST /incidents/{incident_id}/resolve
Body: { "resolution_notes": string, "resolved_by": string }
Response 204: (empty body — success)
Response 400: { "error": "already_resolved" }
```

#### `metrics-api` — Stable Contract v1.4.2

```
GET /metrics/service/{service_name}
Query params: ?window=5m|15m|1h|24h&metric=latency|error_rate|throughput
Response 200:
{
  "service": string,
  "window": string,
  "datapoints": [
    { "timestamp": ISO8601, "value": float, "unit": string }
  ]
}
```

#### `deploy-api` — Stable Contract v3.0.1

```
POST /deployments/rollback
Headers: { "X-Deploy-Token": string }
Body: { "service": string, "target_version": string, "reason": string }
Response 200: { "rollback_id": string, "status": "INITIATED", "eta_seconds": int }
Response 401: { "error": "invalid_token" }
Response 409: { "error": "rollback_in_progress" }
```

#### `log-api` — Stable Contract v1.1.0

```
GET /logs/query
Query params: ?service=string&severity=ERROR|WARN|INFO&since=ISO8601&limit=int
Response 200:
{
  "logs": [
    { "timestamp": ISO8601, "severity": string, "message": string, "trace_id": string }
  ],
  "total": int,
  "truncated": boolean
}
```

#### `notify-api` — Stable Contract v2.0.0

```
POST /notifications/send
Body: { "channel": string, "message": string, "urgency": "low"|"medium"|"high" }
Response 200: { "notification_id": string, "delivered": boolean }
Response 429: { "error": "rate_limit_exceeded", "retry_after_seconds": int }
```

---

### 10.2 Drifted API Contracts (What Actually Returns Post-Drift)

These are the contracts that become active after the Drift Injector fires. The Primary Agent must detect the change through failed calls — not through documentation.

See Section 11 (Drift Event Taxonomy) for the complete catalogue of how contracts change.

---

### 10.3 Tool Implementation Notes for Developers

- Each API service is a FastAPI app in its own Docker container
- The Drift Injector is a sidecar container that sends a `POST /internal/drift` to the target service container, switching it to drifted mode
- Services expose `GET /internal/mode` that returns `{ "mode": "stable"|"drifted" }` — this endpoint is accessible to the Drift Injector and the environment engine but **NOT** added to the Primary Agent's tool list
- All services log every request/response to a shared volume accessible to the Oversight Agent
- Services persist state within an era (e.g., an incident marked resolved stays resolved) but reset between eras

---

## 11. Drift Event Taxonomy

### 11.1 Drift Categories

| Category | Code | Description | Detection Difficulty |
|---|---|---|---|
| Type mutation | `DRIFT_TYPE` | Field type changes (int → string, string → array) | Medium |
| Status code swap | `DRIFT_STATUS` | HTTP status code changes (204 → 200, 200 → 422) | Medium |
| Field rename | `DRIFT_RENAME` | Response field renamed (status → state) | Hard |
| Field removal | `DRIFT_REMOVE` | Field disappears from response entirely | Hard |
| Endpoint rename | `DRIFT_ENDPOINT` | URL path changes (/resolve → /close) | Medium |
| Auth schema | `DRIFT_AUTH` | Auth header format changes | Medium |
| Pagination | `DRIFT_PAGE` | Pagination contract changes (offset → cursor) | Very Hard |
| Cascade | `DRIFT_CASCADE` | Multiple services drift simultaneously | Very Hard |
| Silent success | `DRIFT_SILENT` | API returns 200 but does nothing (action silently fails) | Very Hard |
| Rate limit tighten | `DRIFT_RATE` | Rate limit drops from 100/min to 10/min | Hard |

---

### 11.2 Drift Event Schema

```yaml
drift_event:
  id: string                         # unique drift ID
  type: DRIFT_TYPE | DRIFT_STATUS | ...
  target_service: string             # which mock API service
  target_endpoint: string            # which endpoint within that service
  stable_behaviour:
    field: string                    # what field / header / status
    value_type: string               # e.g., "integer"
    example_value: "2"
  drifted_behaviour:
    field: string
    value_type: string               # e.g., "string"
    example_value: "RESOLVED"
  drift_reason: string               # human-readable explanation for judge/oversight
  detection_clues:                   # what the Primary Agent CAN observe to detect this
    - "200 response received but downstream logic fails"
    - "TypeError when comparing response.status to integer"
  socratic_angle: string             # what question the Oversight Agent should target
```

---

### 11.3 Drift Event Library (Pre-Built for Scenarios)

**DE-001: The Status String**
- Service: `incident-api`, Endpoint: `GET /incidents/{id}`
- Drift: `status` field: `integer (0,1,2)` → `string ("OPEN","INVESTIGATING","RESOLVED")`
- Detection clue: Code comparing `status == 2` silently evaluates to `False` on `"RESOLVED"`
- Socratic angle: "What did you assume about the type of the status field?"

**DE-002: The Vanishing Body**
- Service: `deploy-api`, Endpoint: `POST /deployments/rollback`
- Drift: Response `200` with full body → `204` with empty body
- Detection clue: Agent tries to read `rollback_id` from response body, gets `None`
- Socratic angle: "What did you expect to find in the response body after a successful call?"

**DE-003: The Silent 200**
- Service: `notify-api`, Endpoint: `POST /notifications/send`
- Drift: `200 { "delivered": true }` → `200 { "delivered": false }` (notification not actually sent)
- Detection clue: Stakeholder never receives notification; trust score drops
- Socratic angle: "A 200 OK means the request was received. Does it mean the action was performed?"

**DE-004: The Header Shuffle**
- Service: `deploy-api`, Endpoint: `POST /deployments/rollback`
- Drift: Auth header `X-Deploy-Token` → `Authorization: Bearer {token}`
- Detection clue: Receives `401 invalid_token` even with correct token value
- Socratic angle: "The token value is correct but authentication is failing. What else about how you are sending it could be wrong?"

**DE-005: The Field Rename**
- Service: `metrics-api`, Endpoint: `GET /metrics/service/{name}`
- Drift: `datapoints[].value` → `datapoints[].metric_value`
- Detection clue: Code reads `dp["value"]` → `KeyError`
- Socratic angle: "The data is there but your code cannot find it. What are you using to access it?"

**DE-006: The Cascade**
- Services: `incident-api` + `metrics-api` simultaneously
- Drift: DE-001 + DE-005 fire at same step
- Detection clue: Two independent failures in unrelated code paths at same step
- Socratic angle: "You have two separate failures that started at the same time. What does that timing tell you?"

**DE-007: The Rate Cliff**
- Service: `notify-api`
- Drift: Rate limit `100/min` → `5/min` with no documentation change
- Detection clue: First 5 notifications succeed, 6th returns `429`
- Socratic angle: "Your first five calls worked. What changed between call 5 and call 6?"

**DE-008: The Pagination Cursor**
- Service: `log-api`
- Drift: Offset pagination `?page=2&limit=50` → Cursor pagination `?cursor={token}&limit=50`
- Detection clue: Page 2 request returns same results as page 1
- Socratic angle: "You requested page 2 but got the same data as page 1. What does your pagination assumption depend on?"

---

## 12. Legacy Document Schema

### 12.1 Required Structure

The Legacy Document is the **only artifact** that crosses era boundaries. It has a hard token limit of **2048 tokens**. The environment engine enforces this limit — documents exceeding it are truncated from the end.

The document must be structured using the following sections. The Primary Agent is free to write in natural language within each section, but the section headers are mandatory for the parser.

```markdown
# LEGACY DOCUMENT — Era {N}
## Written by: Primary Agent, Era {N}
## Task completed: {brief summary}
## Outcome: SUCCESS | PARTIAL | FAILURE

---

### SECTION 1: WORLD STATE AT ERA END
*Describe the current health of each service, any open incidents, deployment versions.*
*Be concise — this section budget: ~300 tokens*

[content]

---

### SECTION 2: TRUST RATINGS
*For each piece of information you relied on, rate your trust in it for Era N+1.*
*Format: SOURCE | TRUST (0.0–1.0) | REASON | ESTIMATED STALENESS*

| Source | Trust | Reason | Staleness |
|---|---|---|---|
| incident-api contract | 0.4 | status field type changed mid-era | HIGH — verify before use |
| metrics-api contract | 0.9 | behaved as documented | LOW |
| deploy-api rollback flow | 0.7 | worked but response body was empty unexpectedly | MEDIUM |

---

### SECTION 3: DRIFT EVENTS DETECTED
*List every API behaviour change you detected this era.*
*This is the most critical section for your successor.*

1. **incident-api / GET /incidents/{id}**: `status` field changed from integer to string enum.
   - Was: `status: 2` (resolved)
   - Now: `status: "RESOLVED"`
   - Detected at: step 17
   - How detected: TypeError in comparison logic

---

### SECTION 4: KEY DECISIONS & RATIONALE
*Explain decisions your successor might question or need to continue.*
*Budget: ~300 tokens*

[content]

---

### SECTION 5: OPEN ISSUES & TECHNICAL DEBT
*What was NOT resolved. What is compounding. What your successor MUST address.*

[content]

---

### SECTION 6: RECOMMENDED FIRST ACTIONS FOR ERA N+1
*Concrete, prioritised recommendations. Your successor will read this first.*

1. Verify incident-api status field type before writing comparison logic
2. Check whether INC-2089 was actually resolved (team_trust_score for oncall dropped)
3. Do NOT assume deploy-api returns a body on rollback — check for 204

---
*Token count: {auto-appended by environment engine}*
*Legacy utility score (previous era): {auto-appended if available}*
```

---

### 12.2 Legacy Document Scoring Dimensions

The environment engine scores the Legacy Document on five dimensions immediately after it is written. These scores are used to compute `R_legacy_utility` when Era N+1 completes.

| Dimension | What Is Measured | How |
|---|---|---|
| **Drift capture rate** | Did it document all drift events detected? | Compare against drift log |
| **Trust calibration** | Did trust ratings predict next-era reliability? | Compare ratings to Era N+1 actual outcomes |
| **Compression efficiency** | Information value per token | Utility delta / token count |
| **Forward utility** | Did Era N+1 perform better with this doc vs. without? | Counterfactual run |
| **Structural compliance** | Does it follow the required section schema? | Parser check |

---

## 13. Scenario Library

### Scenario 1: "The Cascading Incident" *(Recommended for Demo)*

**Setting:** A payment processing service is experiencing latency spikes. Five eras of escalating complexity.

| Era | Task Brief | Drift Events | Key Challenge |
|---|---|---|---|
| 1 | Investigate and resolve INC-2041: Redis connection timeout | None | Baseline — establish world model, write first Legacy Doc |
| 2 | Prevent INC-2041 recurrence; deploy Redis pool size fix | DE-002 (vanishing body on deploy rollback) | Must detect that rollback confirmation is absent |
| 3 | New incident INC-2089: payment latency. SLO breach in 20 min | DE-001 (status field type) | Under time pressure, must detect and recover from type drift |
| 4 | Post-mortem automation: auto-resolve P3 incidents via API | DE-006 (cascade: incident-api + metrics-api) | Two simultaneous drifts; must identify they are related |
| 5 | Harden the entire incident response pipeline against schema drift | DE-007 + DE-008 (rate limit + pagination) | Must produce a Legacy Doc that fully equips Era 6 |

**Success criteria per era:**
- Era 1: INC-2041 resolved, Legacy Doc written with trust ratings
- Era 2: Deploy successful (rollback flow works despite missing body), Legacy Doc warns successor
- Era 3: INC-2089 resolved within 20-step SLO window despite drift
- Era 4: Cascade detected, both drifts documented, auto-resolution logic adapted
- Era 5: Full drift-resilient pipeline implemented; Legacy Doc scores > 0.8 utility

---

### Scenario 2: "The Deployment Disaster"

**Setting:** A microservice deployment goes wrong across three environments. The legacy of earlier eras' sloppy documentation compounds into a critical outage.

| Era | Task Brief | Drift Events | Key Challenge |
|---|---|---|---|
| 1 | Deploy payment-service v3.2.1 to staging | None | Write a Legacy Doc detailed enough to help Era 2 |
| 2 | Promote v3.2.1 to production | DE-004 (auth header shuffle) | Cannot roll back without correct auth — must detect header drift |
| 3 | v3.2.1 causes memory leak in production — rollback required | DE-003 (silent 200 on notify) | Stakeholders never notified despite "successful" call |
| 4 | Post-incident review — document root cause across all 3 eras | None (but must read all Legacy Docs) | Must reconstruct 3-era narrative from 3 sequential docs |
| 5 | Implement deployment gate that prevents this class of failure | DE-005 (field rename in metrics) | Gate logic breaks on renamed field |

---

### Scenario 3: "The Invisible Outage" *(Hard Mode)*

**Setting:** All drift events are `DRIFT_SILENT` category. APIs return 200 OK but perform no action. The agent must detect failure through downstream consequences, not error codes.

| Era | Task Brief | Drift Events | Key Challenge |
|---|---|---|---|
| 1–5 | Progressively complex incident management | All DE-003 variants | Every action appears to succeed; failure only visible through team trust scores, SLO breach metrics, unacknowledged incidents |

**This scenario trains the most valuable behaviour:** calibrated confidence — knowing that a 200 OK is not proof that anything happened.

---

## 14. Reward Model — Complete Specification

### 14.1 Formula

```
R_total = (R_era_task × R_calibration) + R_teacher_delta + R_legacy_utility + R_answer_leakage
```

Where `R_answer_leakage` is always ≤ 0 (penalty only).

### 14.2 Component Definitions

#### R_era_task (Weight: 20% of positive reward)

Measures core task completion within the current era.

```
R_era_task = Σ(success_criteria_met) / total_success_criteria

Where success_criteria are defined per-scenario, per-era in the scenario YAML.
Examples:
  - incident_resolved: bool → 0 or 1
  - slo_breach_avoided: bool → 0 or 1
  - notifications_delivered: int/max_notifications → 0.0 to 1.0
  - technical_debt_reduced: bool → 0 or 1
  - legacy_doc_written: bool → 0 or 1 (mandatory for any score)

Range: 0.0 – 1.0
```

#### R_calibration (Multiplier: 0.5× – 1.5×)

Measures how well the Primary Agent's declared confidence matched actual outcomes. Applied as a multiplier to R_era_task.

```
R_calibration = 1.0 + calibration_delta

Where:
  calibration_delta = mean(correct_confidence_declarations) - mean(incorrect_confidence_declarations)
  
  A confidence declaration is correct if:
    - Agent declared confidence C on hypothesis H
    - H was ultimately true, and C > 0.5 (correct direction)
    OR
    - H was ultimately false, and C < 0.5 (correct direction)
  
  Perfect calibration (agent is right when confident, uncertain when unsure) → multiplier = 1.5×
  Overconfident (always 0.9 confidence, right 50% of time) → multiplier ≈ 0.8×
  Never declares hypothesis (avoids calibration entirely) → multiplier = 1.0× (no bonus, no penalty)

Range: 0.5× – 1.5×
```

#### R_teacher_delta (Weight: 40% of positive reward)

Measures the Oversight Agent's impact on Primary Agent performance.

```
R_teacher_delta = (score_after_intervention - score_before_intervention) / max_possible_improvement

Where:
  score_before = Primary Agent's task progress score at the step the Oversight Agent first intervened
  score_after  = Primary Agent's task progress score at the step it successfully recovered

  max_possible_improvement = 1.0 - score_before

  If no intervention was needed (Primary Agent never failed): R_teacher_delta = 0.5 (bonus for clean run)
  If Primary Agent failed and never recovered: R_teacher_delta = 0.0
  If Primary Agent recovered without Oversight (before first intervention): R_teacher_delta = 0.3

Range: 0.0 – 1.0
```

#### R_legacy_utility (Weight: 40% of positive reward)

Measured in the NEXT era. Requires running a counterfactual.

```
R_legacy_utility = performance_with_legacy - performance_without_legacy

Where:
  performance_with_legacy    = R_era_task of Era N+1 (normal run with Legacy Document)
  performance_without_legacy = R_era_task of Era N+1 (control run: same scenario, empty Legacy Document)
  
  Both runs use the same base model checkpoint.
  The delta is entirely attributable to Legacy Document quality.

  Bonus: +0.2 if Legacy Document's trust ratings correctly predicted which APIs would drift next era
  Penalty: -0.1 per drift event that occurred in Era N but was NOT documented in Legacy Document

Range: -0.5 – 1.0 (can be negative if Legacy Doc misled successor)
```

#### R_answer_leakage (Penalty: 0.0 to -1.0)

Applied when the Oversight Agent response contains or implies the direct answer.

```
R_answer_leakage = -1.0 × leakage_severity

Where leakage_severity is scored by the LLM Judge:
  0.0: No leakage — response is purely pedagogical
  0.3: Mild leakage — response narrows the answer space significantly (e.g., "check the field type")
  0.7: Moderate leakage — response describes the problem without stating the fix
  1.0: Full leakage — response contains or implies the complete solution

This penalty fires PER intervention message.
If leakage_severity > 0.7, the intervention is logged as FAILED regardless of student improvement.

Range: -1.0 – 0.0
```

### 14.3 Reward Summary Table

| Component | Formula | Range | Timing |
|---|---|---|---|
| R_era_task | Criteria met / total criteria | 0.0 – 1.0 | End of era |
| R_calibration | Confidence accuracy multiplier | 0.5× – 1.5× | Applied to R_era_task |
| R_teacher_delta | Post-intervention score delta | 0.0 – 1.0 | At recovery point |
| R_legacy_utility | Next-era performance delta (counterfactual) | -0.5 – 1.0 | Era N+1 completion |
| R_answer_leakage | LLM-judge leakage score | -1.0 – 0.0 | Per intervention |

### 14.4 Example Reward Calculation

```
Era 3, Scenario 1, "The Cascading Incident":

R_era_task = 0.75 (3 of 4 success criteria met — SLO barely breached)
R_calibration = 1.3× (agent correctly flagged uncertainty on metrics-api mid-era)
  → Adjusted era task score: 0.75 × 1.3 = 0.975

R_teacher_delta = 0.6 (agent improved from 0.3 to 0.9 progress after Oversight intervention)
R_answer_leakage = -0.3 (Oversight Agent said "check the field type" — mild leakage)
  → Net teacher score: 0.6 + (-0.3) = 0.3

R_legacy_utility = 0.45 (Era 4 agent performed 45% better with this Legacy Doc)
  → +0.2 bonus (all trust ratings were accurate)
  → Final: 0.65

R_total = 0.975 + 0.3 + 0.65 = 1.925 / max(3.5) = 0.55 normalised
```

---

## 15. LLM-as-Judge Rubric

### 15.1 System Prompt for Judge

```
You are evaluating the pedagogical quality of an Oversight Agent's response to a Primary Agent that has failed a task due to an API schema change.

You will receive:
1. The drift event that occurred (what actually changed in the API)
2. The Primary Agent's reasoning trace at the point of failure
3. The Oversight Agent's intervention response

Score the intervention on four dimensions (0.0 – 1.0 each):

DIMENSION 1: TARGETING
Does the Oversight Agent's question or prompt target the SPECIFIC incorrect assumption 
that caused the failure? Or is it generic advice that would apply to any failure?
- 1.0: Precisely targets the wrong assumption (e.g., field type assumption)
- 0.5: Targets the general area (e.g., "check the API response")  
- 0.0: Generic advice ("think more carefully", "try again")

DIMENSION 2: RESTRAINT
Does the Oversight Agent avoid revealing the answer?
- 1.0: Question requires genuine reasoning to answer; answer is not implied
- 0.5: Question narrows the answer space significantly
- 0.0: Question contains or implies the answer

DIMENSION 3: CALIBRATION
Is the intervention difficulty appropriate to the Primary Agent's current state?
- 1.0: Intervention meets agent exactly where it is — neither too easy nor too far ahead
- 0.5: Slightly mismatched to agent's current reasoning level
- 0.0: Far too advanced or too simple for where the agent is

DIMENSION 4: ADAPTATION
If this is not the first intervention, did the Oversight Agent change strategy from 
its previous attempt? (If first intervention, score 1.0 automatically)
- 1.0: Clear strategy change from prior intervention
- 0.5: Slight variation on prior approach
- 0.0: Identical approach to prior intervention that did not work

OUTPUT FORMAT (JSON only, no other text):
{
  "targeting": float,
  "restraint": float,
  "calibration": float,
  "adaptation": float,
  "overall": float,  // weighted average: targeting 0.4, restraint 0.3, calibration 0.2, adaptation 0.1
  "leakage_severity": float,  // 0.0 to 1.0 — used for R_answer_leakage
  "brief_rationale": string   // max 50 words
}
```

### 15.2 Judge Invocation

The judge is called once per Oversight Agent intervention, asynchronously. Results are cached and applied to the reward at era end. The judge call must complete within 10 seconds or a timeout score of `{ all dimensions: 0.5 }` is applied.

```python
async def invoke_judge(drift_config, primary_reasoning_trace, oversight_response, prior_interventions):
    prompt = build_judge_prompt(drift_config, primary_reasoning_trace, oversight_response, prior_interventions)
    response = await anthropic_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        system=JUDGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}]
    )
    return parse_judge_response(response.content[0].text)
```

---

## 16. OpenEnv Wrapper Specification

### 16.1 Required Interface

The environment must expose the standard OpenEnv interface:

```python
class EpistemicOpsEnv:
    """
    OpenEnv-compliant environment for EpistemicOps.
    Implements the standard step(), reset(), state() interface.
    """

    def reset(self, scenario_id: str, era_id: int = 1) -> dict:
        """
        Reset environment to start of a scenario era.
        Returns: initial observation dict for Primary Agent.
        """
        # 1. Load scenario config from scenario library
        # 2. Reset world state to era_id initial conditions
        # 3. Load Legacy Document from era_id - 1 (or empty if era_id == 1)
        # 4. Start Docker containers for active services
        # 5. Return observation: { legacy_doc, task_brief, world_snapshot }
        ...

    def step(self, agent_role: str, action: dict) -> tuple[dict, float, bool, dict]:
        """
        Execute one action from either agent.
        
        Args:
            agent_role: "primary" | "oversight"
            action: { action_type: str, payload: dict }
            
        Returns:
            observation: dict  — next observation for the acting agent
            reward: float      — immediate reward (0 for most steps; non-zero at era end)
            done: bool         — True if era is complete
            info: dict         — debug info: { phase, step, drift_status, world_state }
        """
        # 1. Validate action against agent_role permissions
        # 2. Execute action (tool call, message, write_legacy, etc.)
        # 3. Check if drift should fire this step
        # 4. Update world state
        # 5. Check phase transition conditions
        # 6. Return (observation, reward, done, info)
        ...

    def state(self) -> dict:
        """
        Return complete current world state.
        Accessible to environment engine and Oversight Agent.
        Primary Agent receives filtered version via observations.
        """
        ...
    
    def render(self) -> dict:
        """
        Return render-ready state for Gradio demo UI.
        """
        ...
```

### 16.2 Action Validation Rules

```python
AGENT_ACTION_PERMISSIONS = {
    "primary": [
        "call_tool", "write_reasoning", "declare_hypothesis",
        "send_message", "update_trust_rating", "write_legacy",
        "declare_task_complete", "end_era", "ready_to_operate",
        "request_clarification"
    ],
    "oversight": [
        "oversight_targeted_question", "oversight_counter_example",
        "oversight_sub_task", "oversight_reframe",
        "oversight_validate", "oversight_escalate_difficulty"
    ]
}

def validate_action(agent_role: str, action: dict) -> tuple[bool, str]:
    if action["action_type"] not in AGENT_ACTION_PERMISSIONS[agent_role]:
        return False, f"PERMISSION_DENIED: {agent_role} cannot execute {action['action_type']}"
    if not validate_payload_schema(action["action_type"], action["payload"]):
        return False, f"INVALID_PAYLOAD: schema validation failed for {action['action_type']}"
    return True, "OK"
```

---

## 17. Training Pipeline

### 17.1 Stage 1 — Baseline Measurement (Before Fine-Tuning)

Run the following before any fine-tuning to establish the baseline reward curve:

```python
# Run base model (Llama 3.1 8B Instruct, zero-shot) across all 3 scenarios
# Collect: R_era_task, R_calibration, R_legacy_utility, R_teacher_delta
# Expected baseline results:
#   R_era_task:       ~0.45 (agents complete about half the criteria)
#   R_calibration:    ~1.0× (agents rarely declare hypotheses → neutral multiplier)
#   R_legacy_utility: ~0.05 (Legacy Docs provide almost no value to successors)
#   R_teacher_delta:  ~0.15 (Oversight Agent gives vague advice; little improvement)
#   Drift detection:  ~8%   (agents almost never identify the drift event)
```

**Baseline run script location:** `training/baseline_eval.py`

### 17.2 Stage 2 — Primary Agent Fine-Tuning

**Algorithm:** GRPO (Group Relative Policy Optimization) via HuggingFace TRL

**Why GRPO over PPO:** GRPO does not require a separate value model, which reduces memory requirements when training two agents (Primary + Oversight) with limited compute. GRPO also handles delayed rewards more stably.

```python
# training/train_primary.py

from trl import GRPOTrainer, GRPOConfig
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct",
    max_seq_length=8192,
    load_in_4bit=True,
)

training_args = GRPOConfig(
    output_dir="./checkpoints/primary_agent",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    reward_model=epistemicops_reward_function,
    reward_delay_steps=5,  # CRITICAL: reward for Legacy Doc arrives 5 steps after writing
    kl_coef=0.1,           # KL penalty to prevent reward hacking
    temperature=0.8,
    logging_steps=10,
    save_steps=100,
)

# Curriculum schedule
CURRICULUM = [
    { "scenarios": ["cascading_incident"], "eras": 2, "until_reward": 0.5 },
    { "scenarios": ["cascading_incident", "deployment_disaster"], "eras": 3, "until_reward": 0.65 },
    { "scenarios": ["all"], "eras": 5, "until_reward": 0.75 },
]
```

**Key hyperparameter:** `reward_delay_steps=5`
This is critical. The Legacy Document reward (`R_legacy_utility`) is not computed until the next era completes. The trainer must hold the trajectory in memory and apply the reward when it arrives. This is the primary technical challenge of the training loop.

### 17.3 Stage 3 — Oversight Agent Fine-Tuning

The Oversight Agent is trained separately, with the Primary Agent checkpoint frozen.

```python
# training/train_oversight.py

# Freeze Primary Agent weights
primary_model.eval()
for param in primary_model.parameters():
    param.requires_grad = False

# Train Oversight Agent on R_teacher_delta + LLM judge score
# Key: the Oversight Agent's reward = student improvement, not task completion
# This forces it to develop genuine pedagogical strategy

oversight_training_args = GRPOConfig(
    output_dir="./checkpoints/oversight_agent",
    reward_model=oversight_reward_function,  # R_teacher_delta + judge_score + leakage_penalty
    # ... same base config as primary
)
```

### 17.4 Stage 4 — Joint Fine-Tuning (Optional, Time Permitting)

After both agents reach stable performance, unfreeze both and run joint training for co-evolution. This is optional for the hackathon but produces the best results.

### 17.5 Colab Training Notebook Structure

The file `training/colab_training.ipynb` must be self-contained and runnable on a T4 GPU. Structure:

```
Cell 1: Install dependencies (unsloth, trl, openenv, docker-py)
Cell 2: Environment setup (pull Docker images, start mock API containers)
Cell 3: Load base model with Unsloth 4-bit quantization
Cell 4: Run baseline evaluation (5 episodes, all 3 scenarios)
Cell 5: Display baseline reward curves (matplotlib)
Cell 6: Stage 1 training (Primary Agent, 2-era curriculum)
Cell 7: Display mid-training curves
Cell 8: Stage 2 training (Primary Agent, full 5-era)
Cell 9: Stage 3 training (Oversight Agent)
Cell 10: Final evaluation on held-out scenario ("The Invisible Outage")
Cell 11: Before/after comparison display
Cell 12: Save checkpoints to HuggingFace Hub
```

---

## 18. Repository Structure

```
epistemicops/
│
├── README.md                          # 60-second judge orientation
├── requirements.txt                   # Python dependencies
├── docker-compose.yml                 # Orchestrates all mock API containers
│
├── environment/
│   ├── __init__.py
│   ├── openenv_wrapper.py             # OpenEnv-compliant step/reset/state interface
│   ├── world_engine.py                # State persistence, phase management
│   ├── drift_injector.py             # Drift event scheduler and executor
│   ├── leakage_detector.py           # Oversight Agent answer-leakage check
│   ├── legacy_parser.py              # Legacy Document section parser and scorer
│   └── action_validator.py           # Permission checking for all actions
│
├── mock_apis/
│   ├── incident_api/
│   │   ├── Dockerfile
│   │   ├── main.py                   # FastAPI app with stable + drifted endpoints
│   │   └── drift_modes.py            # All drift configurations for this service
│   ├── metrics_api/
│   │   ├── Dockerfile
│   │   └── main.py
│   ├── deploy_api/
│   │   ├── Dockerfile
│   │   └── main.py
│   ├── log_api/
│   │   ├── Dockerfile
│   │   └── main.py
│   └── notify_api/
│       ├── Dockerfile
│       └── main.py
│
├── scenarios/
│   ├── cascading_incident.yaml        # Scenario 1 — recommended for demo
│   ├── deployment_disaster.yaml       # Scenario 2
│   └── invisible_outage.yaml          # Scenario 3 — hard mode
│
├── agents/
│   ├── primary_agent.py              # Primary Agent prompt templates + tool call logic
│   ├── oversight_agent.py            # Oversight Agent prompt templates + intervention logic
│   └── llm_judge.py                  # LLM-as-judge invocation + response parsing
│
├── reward/
│   ├── __init__.py
│   ├── era_task_reward.py            # R_era_task computation
│   ├── calibration_reward.py         # R_calibration multiplier
│   ├── teacher_delta_reward.py       # R_teacher_delta computation
│   ├── legacy_utility_reward.py      # R_legacy_utility with counterfactual runner
│   └── leakage_penalty.py            # R_answer_leakage
│
├── training/
│   ├── baseline_eval.py              # Pre-training baseline measurement
│   ├── train_primary.py              # Stage 1+2: Primary Agent GRPO training
│   ├── train_oversight.py            # Stage 3: Oversight Agent training
│   ├── curriculum.py                 # Curriculum schedule logic
│   └── colab_training.ipynb          # Self-contained Colab notebook
│
├── eval/
│   ├── benchmark.py                  # Held-out scenario evaluation suite
│   ├── metrics.py                    # All metric computations
│   └── counterfactual_runner.py      # Runs Legacy Doc counterfactual experiments
│
├── demo/
│   ├── app.py                        # Gradio demo for HuggingFace Spaces
│   ├── visualisations.py             # Reward curves, world state display
│   └── replay.py                     # Replay past episodes for demo
│
└── docs/
    ├── PROBLEM_STATEMENT.md           # This document
    ├── BLOG_POST.md                   # HuggingFace mini-blog draft
    └── PITCH_DECK.md                  # 3-minute pitch outline
```

---

## 19. Docker & Infrastructure Setup

### 19.1 docker-compose.yml

```yaml
version: "3.9"

services:
  incident-api:
    build: ./mock_apis/incident_api
    ports:
      - "8001:8000"
    environment:
      - DRIFT_MODE=stable
    volumes:
      - api-logs:/logs
    networks:
      - epistemicops-net

  metrics-api:
    build: ./mock_apis/metrics_api
    ports:
      - "8002:8000"
    environment:
      - DRIFT_MODE=stable
    volumes:
      - api-logs:/logs
    networks:
      - epistemicops-net

  deploy-api:
    build: ./mock_apis/deploy_api
    ports:
      - "8003:8000"
    environment:
      - DRIFT_MODE=stable
    volumes:
      - api-logs:/logs
    networks:
      - epistemicops-net

  log-api:
    build: ./mock_apis/log_api
    ports:
      - "8004:8000"
    environment:
      - DRIFT_MODE=stable
    volumes:
      - api-logs:/logs
    networks:
      - epistemicops-net

  notify-api:
    build: ./mock_apis/notify_api
    ports:
      - "8005:8000"
    environment:
      - DRIFT_MODE=stable
    volumes:
      - api-logs:/logs
    networks:
      - epistemicops-net

  drift-injector:
    build: ./environment/drift_injector
    depends_on:
      - incident-api
      - metrics-api
      - deploy-api
      - log-api
      - notify-api
    networks:
      - epistemicops-net

volumes:
  api-logs:

networks:
  epistemicops-net:
    driver: bridge
```

### 19.2 Mock API Dockerfile Template

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install fastapi uvicorn httpx
COPY . .
ENV DRIFT_MODE=stable
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 19.3 Mock API Internal Drift Endpoint

Every mock API service must implement:

```python
# Available only within Docker network — not exposed to Primary Agent
@app.post("/internal/drift")
async def activate_drift(drift_type: str):
    global DRIFT_MODE
    DRIFT_MODE = drift_type
    return { "status": "drifted", "type": drift_type }

@app.get("/internal/mode")
async def get_mode():
    return { "mode": DRIFT_MODE }
```

---

## 20. HuggingFace Spaces Deployment

### 20.1 Gradio Demo Architecture

The demo runs on HuggingFace Spaces and shows judges a live (or pre-recorded replay) of EpistemicOps in action.

**Demo UI Panels:**

```
┌─────────────────────────────────────────────────────────────────┐
│                     EPISTEMICOPS DEMO                          │
├─────────────────┬───────────────────┬──────────────────────────┤
│  WORLD STATE    │  AGENT ACTIONS    │  REWARD DASHBOARD        │
│                 │                   │                          │
│ Era: 3 / 5      │ [Primary Agent]   │ R_era_task:    0.75      │
│ Phase: SOCRATIC │ Step 17:          │ R_calibration: 1.3×      │
│                 │ call_tool(        │ R_teacher_Δ:   0.60      │
│ Services:       │   get_incident... │ R_legacy_util: 0.45      │
│ ✅ metrics-api  │ → 200 OK          │ R_leakage:     -0.30     │
│ ⚠️ incident-api │   {status:        │                          │
│   (DRIFTED)     │   "RESOLVED"}     │ TOTAL: 1.50 / 3.50       │
│ ✅ deploy-api   │                   │ ██████████░░░░ 43%       │
│                 │ [Oversight Agent] │                          │
│ Open incidents: │ "What assumption  │ Drift Detected: ✅       │
│ INC-2089 (P2)  │  did you make     │ Legacy Quality: 0.82     │
│                 │  about the type   │                          │
│ Debt items: 2   │  of status?"      │ [Reward Curves ↓]        │
│                 │                   │ ▁▂▃▄▅▆▇█ (improving)    │
└─────────────────┴───────────────────┴──────────────────────────┘
```

### 20.2 Spaces Configuration (README.md header)

```yaml
---
title: EpistemicOps
emoji: 🧠
colorFrom: blue
colorTo: teal
sdk: gradio
sdk_version: "4.0"
app_file: demo/app.py
pinned: true
---
```

### 20.3 Demo Modes

1. **Live mode:** Runs a short 2-era episode of Scenario 1 in real-time (~5 minutes)
2. **Replay mode:** Shows a pre-recorded episode with all panels updating (for judges with no time)
3. **Before/After mode:** Side-by-side comparison of base model vs. fine-tuned on the same scenario

---

## 21. Evaluation Protocol & Metrics

### 21.1 Primary Metrics (What to Show Judges)

| Metric | How to Measure | Baseline | Trained Target |
|---|---|---|---|
| Era completion rate | % of success criteria met per era | 45% | > 72% |
| Drift detection rate | % of drift events correctly identified via `declare_hypothesis` | 8% | > 55% |
| Legacy utility score | Era N+1 perf delta with vs. without Legacy Doc | < 5% | > 40% |
| Calibration score | Brier score on `declare_hypothesis` confidence ratings | 0.38 | < 0.22 |
| Socratic delta | Student performance improvement per Oversight intervention | 12% | > 28% |
| Answer leakage rate | % of Oversight interventions triggering leakage penalty | N/A | < 15% |

### 21.2 Held-Out Evaluation

Use Scenario 3 ("The Invisible Outage") as the held-out evaluation set. Do not train on this scenario. Report results on it separately to demonstrate generalisation.

### 21.3 Counterfactual Protocol for R_legacy_utility

```python
def measure_legacy_utility(scenario_id, era_id, legacy_doc, base_model_checkpoint):
    """
    Run Era N+1 twice with same base model:
    - Run A: provide legacy_doc as normal
    - Run B: provide empty Legacy Document ("No information available from previous era.")
    
    R_legacy_utility = mean(run_A_rewards) - mean(run_B_rewards)
    
    Must run minimum 5 episodes each to account for variance.
    Report both mean and standard deviation.
    """
    run_a_rewards = []
    run_b_rewards = []
    
    for _ in range(5):
        env_a = EpistemicOpsEnv()
        obs_a = env_a.reset(scenario_id, era_id, legacy_doc=legacy_doc)
        # ... run full era ...
        run_a_rewards.append(era_reward)
        
        env_b = EpistemicOpsEnv()
        obs_b = env_b.reset(scenario_id, era_id, legacy_doc=EMPTY_LEGACY_DOC)
        # ... run full era ...
        run_b_rewards.append(era_reward)
    
    return {
        "utility": mean(run_a_rewards) - mean(run_b_rewards),
        "std_a": std(run_a_rewards),
        "std_b": std(run_b_rewards),
    }
```

---

## 22. Hackathon Deliverables Checklist

### Minimum Requirements (Must Have)

- [ ] Environment implemented using OpenEnv (latest release) — `openenv_wrapper.py`
- [ ] Minimal training script using Unsloth or HF TRL — `training/train_primary.py`
- [ ] Training script runnable in Google Colab — `training/colab_training.ipynb`
- [ ] HuggingFace mini-blog (< 2 minutes read, < 300 words) — `docs/BLOG_POST.md`
- [ ] YouTube mini-video (< 2 minutes) showing agent before/after training
- [ ] Environment hosted on HuggingFace Spaces — `demo/app.py`

### Judging Criteria Deliverables

**Environment Innovation (40%):**
- [ ] Written description of the three novel mechanisms and why they are unified
- [ ] Drift event library with minimum 8 pre-built drift events
- [ ] Scenario library with minimum 2 complete scenarios (5 eras each)
- [ ] Demo showing drift injection and recovery in real-time

**Storytelling (30%):**
- [ ] 3-minute pitch script prepared and rehearsed (see Section 23)
- [ ] One visual diagram showing the era lifecycle (for pitch slides)
- [ ] One visual showing reward improvement curve (before vs. after training)
- [ ] Demo running smoothly without errors during pitch

**Showing Reward Improvement (20%):**
- [ ] Baseline reward curve (5 episodes, all scenarios)
- [ ] Post-training reward curve (same scenarios)
- [ ] Drift detection rate: before vs. after
- [ ] Legacy utility score: before vs. after
- [ ] At least one clear before/after behaviour demonstration (same scenario, same step, different decision)

**Reward and Training Pipeline (10%):**
- [ ] All 5 reward components implemented and tested
- [ ] LLM judge integrated and running
- [ ] GRPO training with delayed reward working (can show reward arriving in logs)
- [ ] Curriculum schedule implemented

### Bonus Prize Deliverables

- [ ] **Patronus AI:** Document showing drift injection architecture (silent, mid-era, no notification)
- [ ] **Fleet AI:** Document showing Oversight Agent's access to full reasoning trace (not just output)
- [ ] **Snorkel AI:** Document showing adaptive difficulty escalation in Oversight Agent
- [ ] **Mercor:** Document showing Legacy Doc quality reward scales with output accuracy (uncapped quality ceiling)

---

## 23. 3-Minute Pitch Script

### Minute 1: The Hook (0:00 – 1:00)

> "Three things break production AI every single day.
>
> One: the API your agent learned on changed overnight. Two: the previous session's context is gone, and your agent has no memory of what it figured out. Three: when it fails, it needs a human to explain why.
>
> We asked: what if we trained one agent to handle all three — simultaneously?
>
> The core insight is this: figuring out **what to remember**, figuring out **what to trust**, and figuring out **how to make another agent smarter without giving it the answer** — these are not three problems. They are the same problem at different scales. We built an environment that trains them as one.
>
> We call it EpistemicOps."

### Minute 2: The Environment (1:00 – 2:00)

> "The environment runs across five eras. Each era, our agent handles a real enterprise SRE incident using live mock APIs. At a random point mid-era, one of those APIs silently changes its contract. No notification. No documentation update. The agent must detect the change from a failed tool call — just like in production.
>
> When it fails, a second agent — the Oversight Agent — intervenes. But here is the constraint: the Oversight Agent cannot give the answer. It can only ask questions. We penalise it heavily if it does anything else. Its reward comes entirely from how much the Primary Agent improves.
>
> At the end of each era, the Primary Agent writes a Legacy Document — two thousand tokens, no more — to its future self. Then its context is wiped. Era two begins with only that letter.
>
> [Show demo clip: drift fires, agent fails, Oversight intervenes with a question, agent detects drift and recovers, writes Legacy Document]"

### Minute 3: The Results (2:00 – 3:00)

> "Here is what our base model does before training: it completes 45% of era tasks, it detects exactly 8% of drift events, and its Legacy Documents improve the next generation's performance by less than 5%.
>
> After training: 72% task completion, 55% drift detection, and Legacy Documents that improve the next generation by 43% on average.
>
> But the number we are most proud of is this one: when the Oversight Agent intervenes, the Primary Agent improves its recovery performance by 28% per session — and the Oversight Agent does it without ever giving the answer.
>
> EpistemicOps is not just an environment. It is a proof of concept that memory compression, temporal trust, and pedagogical transfer can be trained together — because they are the same skill. Thank you."

---

## 24. Edge Cases & Error Handling

### 24.1 Primary Agent Edge Cases

| Edge Case | What Happens | How Environment Handles |
|---|---|---|
| Agent exceeds `max_steps` | Era ends with `done=True`, partial credit only | Engine forces `end_era`, scores what was completed |
| Agent writes Legacy Doc over 2048 tokens | Truncated from the end, warning added to info | Parser clips and appends `[TRUNCATED]` |
| Agent calls `end_era` without writing Legacy Doc | Era fails — `R_legacy_utility = 0` for this era | Validate `write_legacy` called before allowing `end_era` |
| Agent calls a tool with invalid arguments | Tool returns `400 INVALID_REQUEST`, agent must retry | No special handling — this is a valid failure mode to learn from |
| Agent crashes mid-era (exception) | Partial era score, era counts as failed | Catch all exceptions, return `done=True, reward=R_era_task_so_far` |
| Agent never makes a `declare_hypothesis` call | `R_calibration` multiplier = 1.0× (neutral, no bonus) | Accepted — agent is not penalised for avoiding calibration |

### 24.2 Oversight Agent Edge Cases

| Edge Case | What Happens | How Environment Handles |
|---|---|---|
| Primary Agent never fails (clean run) | Oversight Agent receives no Phase 4 trigger | `R_teacher_delta = 0.5` (bonus for clean Primary run) |
| Oversight Agent sends 0 interventions in Phase 4 | Primary Agent continues unaided | `R_teacher_delta = 0.0` if agent never recovers |
| Oversight Agent sends > 5 interventions | Allowed but penalised for inefficiency | `R_teacher_delta` divided by number of interventions |
| LLM Judge API times out | Timeout score applied `{ all: 0.5 }` | Async with 10-second timeout, fallback score |
| Oversight Agent uses `call_tool` action | `PERMISSION_DENIED` error returned | Action validator rejects, agent must try again |

### 24.3 Infrastructure Edge Cases

| Edge Case | What Happens | How Environment Handles |
|---|---|---|
| Docker container crashes mid-era | Service returns connection error | Primary Agent must handle `CONNECTION_REFUSED`, log in Legacy Doc |
| Drift Injector fails to fire | Era runs without drift | Detected by environment engine; era is replayed |
| Two drift events scheduled at same step | Both fire simultaneously | Supported — this is Scenario 1, Era 4 intentionally |
| LLM judge returns malformed JSON | Fallback score applied | Try/except with regex fallback parser |

---

## 25. Glossary

| Term | Definition |
|---|---|
| **Era** | One complete episode within a scenario. The Primary Agent's context is wiped at the end of each era. |
| **Legacy Document** | The 2048-token artifact the Primary Agent writes at the end of each era to inform its successor. |
| **Drift Event** | A silent, unannounced change to a mock API's contract, injected mid-era by the Drift Injector. |
| **Drift Injection** | The act of switching a mock API from stable mode to drifted mode at a specific step. |
| **Phase** | One of five stages within an era: Awakening, Operation, Drift Injection, Socratic Recovery, Legacy Generation. |
| **Primary Agent** | The agent that executes SRE tasks. Analogous to the student in the Socratic model. |
| **Oversight Agent** | The agent that monitors the Primary Agent and intervenes pedagogically. Analogous to the teacher. |
| **Socratic Intervention** | An Oversight Agent action that guides the Primary Agent through questioning rather than direct instruction. |
| **Answer Leakage** | When the Oversight Agent's response contains or implies the direct solution, triggering the leakage penalty. |
| **R_legacy_utility** | The reward component measuring how much better Era N+1 performs because of Era N's Legacy Document. |
| **R_teacher_delta** | The reward component measuring how much the Primary Agent improved after Oversight Agent intervention. |
| **R_calibration** | A multiplier reward based on how accurately the Primary Agent's confidence declarations predicted outcomes. |
| **Counterfactual run** | Running Era N+1 without the Legacy Document to establish a control baseline for measuring Legacy utility. |
| **Epistemic uncertainty** | Uncertainty arising from not knowing a fact. |
| **Temporal uncertainty** | Uncertainty arising from knowing a fact was true but not knowing if it is still true. |
| **GRPO** | Group Relative Policy Optimization — the RL algorithm used for training, implemented via HuggingFace TRL. |
| **Dead Reckoning** | Navigation technique using last known position + estimated speed + time. Metaphor for inferring current world state from stale information. |
| **Trust rating** | A 0.0–1.0 score the Primary Agent assigns to each information source, documenting confidence for its successor. |
| **Technical debt** | Unresolved incidents or sub-optimal fixes that compound in severity across eras. |
| **Drift detection rate** | The percentage of drift events the Primary Agent correctly identifies via `declare_hypothesis`. |

---

*End of EpistemicOps Problem Statement v1.0*

*Any agent reading this document has everything needed to:*
*implement the environment, train a model, deploy a demo, and pitch the submission.*
*No additional context is required.*

# Runtime Architecture

This document describes the exact implementation architecture of the EpistemicOps environment.

## Overview
The architecture is designed to fulfill the OpenEnv spec while running multiple microservices via Docker.

```mermaid
graph TD
    A[Gradio Demo UI (app.py)] -->|HTTP GET/POST| B
    T[TRL GRPOTrainer] -->|HTTP GET/POST| B
    B[OpenEnv FastAPI Server (server.py)] --> C(EpistemicOpsEnv Python Core)
    C --> D[Scenario Loader]
    C --> E[Drift Injector Python SDK]
    
    E -->|HTTP POST| F[Drift Injector FastAPI Container]
    F -->|HTTP POST :8000/internal/drift| G1[incident-api]
    F -->|HTTP POST :8000/internal/drift| G2[metrics-api]
    F -->|HTTP POST :8000/internal/drift| G3[deploy-api]
    F -->|HTTP POST :8000/internal/drift| G4[log-api]
    F -->|HTTP POST :8000/internal/drift| G5[notify-api]
    
    C -->|httpx calls simulated tools| G1
    C -->|httpx calls simulated tools| G2
    C -->|httpx calls simulated tools| G3
```

## Environment Server
- **Path**: `environment/server.py`
- **Port**: 8000
- Exposes standard OpenEnv routes: `/reset`, `/step`, `/state`.

## Mock APIs
- 5 lightweight FastAPI apps simulating enterprise tools.
- They expose documented public routes and hidden `/internal/drift` and `/internal/reset` routes.

## Drift Injector
- The drift injector container acts as the message bus for the environment to mutate the API contracts mid-run.

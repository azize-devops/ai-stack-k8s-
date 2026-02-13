# AI Agent Workflows

This module provides production-ready AI agent workflows built on top of CrewAI and LangGraph. Both frameworks integrate with OpenAI-compatible APIs, enabling seamless operation against LocalAI (deployed in the Kubernetes cluster) or any external OpenAI-compatible endpoint.

## Overview

The module contains two independent agent systems:

1. **CrewAI Multi-Agent Research Pipeline** -- A three-agent collaborative workflow for automated research, report writing, and quality review.
2. **LangGraph Stateful Workflow** -- A graph-based stateful pipeline with conditional routing that classifies queries and optionally retrieves RAG context before generating responses.

---

## CrewAI Research Pipeline

Located in `crewai-research-agent/`.

### Agents

| Agent | Role | Responsibility |
|-------|------|---------------|
| **Researcher** | Senior Research Analyst | Searches for and gathers comprehensive information on a given topic |
| **Writer** | Technical Writer | Synthesizes research findings into a clear, structured report |
| **Reviewer** | Quality Reviewer | Reviews the report for accuracy, completeness, and clarity |

### How It Works

1. The **Researcher** agent receives a topic and produces detailed research notes.
2. The **Writer** agent takes those notes and drafts a structured technical report.
3. The **Reviewer** agent evaluates the report and provides a final quality assessment.

The agents execute sequentially in a CrewAI `Crew` with a configurable process type (sequential by default).

### Running

```bash
# From the agents/ directory
cd crewai-research-agent/

# With a custom topic
python main.py "Kubernetes autoscaling strategies"

# With the default topic
python main.py
```

Output is saved to `crewai-research-agent/output/` with timestamped filenames.

---

## LangGraph Stateful Workflow

Located in `langgraph-workflow/`.

### Architecture

```
START
  |
  v
classify_query  -->  conditional_route
                        |            |
                    route="rag"   route="direct"
                        |            |
                        v            |
                  retrieve_context   |
                        |            |
                        v            v
                    generate_response
                        |
                        v
                    check_quality
                        |
                        v
                       END
```

### Nodes

| Node | Purpose |
|------|---------|
| `classify_query` | Uses an LLM to classify the incoming query as factual, analytical, or creative |
| `retrieve_context` | Retrieves relevant context from Qdrant (falls back to simulated context if unavailable) |
| `generate_response` | Generates a final answer, optionally augmented with retrieved context |
| `check_quality` | Validates response quality and completeness |

### Running

```bash
cd langgraph-workflow/

# Run with example queries
python main.py

# Run with a custom query
python main.py "What are the benefits of vector databases?"
```

---

## Configuration

Both modules are configured via environment variables. Copy the project-level `.env.example` to `.env` in the `agents/` directory or set the variables in your shell.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BASE_URL` | `http://localai:8080/v1` | Base URL for the OpenAI-compatible API |
| `LLM_API_KEY` | `sk-no-key-required` | API key (LocalAI typically does not require one) |
| `LLM_MODEL_NAME` | `gpt-3.5-turbo` | Model name as recognized by the LLM backend |
| `QDRANT_URL` | `http://qdrant:6333` | Qdrant vector database endpoint |
| `QDRANT_COLLECTION` | `documents` | Qdrant collection name for RAG retrieval |
| `LOG_LEVEL` | `INFO` | Logging verbosity (DEBUG, INFO, WARNING, ERROR) |

### Using with OpenAI

```bash
export LLM_BASE_URL=https://api.openai.com/v1
export LLM_API_KEY=sk-your-openai-key
export LLM_MODEL_NAME=gpt-4
```

### Using with LocalAI (Kubernetes cluster)

```bash
export LLM_BASE_URL=http://localai.ai-stack.svc.cluster.local:8080/v1
export LLM_API_KEY=sk-no-key-required
export LLM_MODEL_NAME=gpt-3.5-turbo
```

---

## Docker

A multi-stage Dockerfile is provided in `docker/` for containerized execution.

```bash
cd docker/
docker build -t ai-agents:latest .
docker run --env-file ../../.env ai-agents:latest
```

---

## Project Structure

```
agents/
├── requirements.txt              # Shared Python dependencies
├── README.md                     # This file
├── crewai-research-agent/
│   ├── config.yaml               # Agent and task configuration
│   ├── agents.py                 # Agent definitions (Researcher, Writer, Reviewer)
│   ├── tasks.py                  # Task definitions (research, write, review)
│   └── main.py                   # CLI entry point for the CrewAI pipeline
├── langgraph-workflow/
│   ├── state.py                  # State schema for the workflow graph
│   ├── nodes.py                  # Node functions (classify, retrieve, generate, check)
│   ├── graph.py                  # StateGraph construction with conditional routing
│   └── main.py                   # CLI entry point for the LangGraph workflow
└── docker/
    ├── Dockerfile                # Multi-stage build for containerized agents
    └── requirements.txt          # Dependencies for Docker build context
```

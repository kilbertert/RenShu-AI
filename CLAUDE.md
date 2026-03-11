# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SmartTCM-Agent-SYSTEM is a multi-agent Traditional Chinese Medicine (TCM) consultation system built with FastAPI (backend), React (frontend), and LangGraph (multi-agent orchestration). The system integrates DeepSeek-TCM and other TCM-tuned LLMs with GraphRAG knowledge retrieval to provide intelligent TCM diagnosis, herbal consultation, prescription recommendations, and wellness guidance.

### Tech Stack

**Backend:**
- FastAPI (web framework)
- LangGraph (multi-agent workflow)
- PostgreSQL (structured data: patients, cases)
- Neo4j (graph data: herbs, syndromes, prescriptions relationships)
- SQLAlchemy (ORM)
- Pydantic (data validation)
- GraphRAG (knowledge retrieval, located in `backend/graphrag/`)

**Frontend:**
- React 19 with TypeScript
- Vite (build tool)
- React Router (routing)
- Lucide React (icons)

**LLM Providers:**
- DeepSeek (default, TCM-tuned) - supports `reasoning_content` format
- OpenAI-compatible APIs - supports `type='reasoning'` format
- Ollama (local deployment)
- Thinking mode supports: DeepSeek, Qwen, GLM (standard), OpenAI gpt-5, Claude, Gemini

---

## Common Commands

### Backend (Python)

```bash
# From project root
cd backend

# Create/activate virtual environment (if needed)
uv venv --python=3.11.7
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

# Install dependencies
uv add . --dev

# Run development server (auto-reload)
python main.py
# Or: python -m uvicorn main:app --reload

# Run with custom port
python -m uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

### Frontend (React)

```bash
# From project root
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

### Database

```bash
# PostgreSQL: Create database manually, then run initialization scripts if available
# Neo4j: Start Neo4j service, then import TCM knowledge graph data
# Note: Database initialization scripts are currently being set up
```

---

## Architecture Overview

### Backend Structure

```
backend/app/src/
├── super_agent/        # Refactored shared modules (replaces duplicates in agent/)
│   ├── utils/          # Shared utilities (estimate_tokens, solar_term, profile formatting)
│   │   ├── tcm_utils.py      # Unified utility functions (replaces 6+ duplicate implementations)
│   │   └── tcm_constants.py  # Unified TCM keywords, solar terms, regions
│   ├── tools/          # Unified tool system
│   │   ├── registry.py       # ToolRegistry singleton (register, discover, get by agent type)
│   │   ├── tcm_tools.py      # Merged @tool functions from 4 files
│   │   └── monitoring.py     # ToolMonitor (call counting, latency, rate limiting)
│   ├── memory/         # 3-layer memory coordination
│   │   └── memory_service.py # MemoryService (Checkpointer + ConversationDB + Mem0)
│   ├── context/        # Focus Sawtooth unified context engine (arXiv:2601.07190)
│   │   ├── focus_engine.py            # Core Sawtooth engine + strategy + summarizer
│   │   └── focus_context_middleware.py # FocusContextMiddleware (P5, replaces 3 middlewares)
│   └── deep_search/    # Unified DeepSearch Agent (custom version only)
│       └── agent.py    # create_deep_search_agent() unified entry point
├── agent/              # Core multi-agent system (LangGraph)
│   ├── tcm_builder.py  # Main graph builder with middleware chain
│   ├── tcm_service.py  # Service layer for agent execution
│   ├── tcm_states.py   # State definitions (TCMAgentState, Input/Output)
│   ├── components/     # Agent components by intent type
│   │   ├── router/     # Intent recognition and routing
│   │   ├── diagnose/   # Diagnosis subgraph (multi-turn, complexity-based)
│   │   ├── wellness/   # Wellness consultation subgraph
│   │   ├── herb/       # Herb consultation handler
│   │   ├── prescription/ # Prescription handler
│   │   └── general/    # General TCM chat handler
│   ├── middleware/     # Middleware chain (guardrails, PII, logging, etc.)
│   └── intent_recognition/ # Intent analysis module
├── controller/         # FastAPI route controllers
│   ├── chat_controller.py        # /api/v1/chat endpoints
│   ├── account_controller.py     # Authentication endpoints
│   ├── model_config_controller.py # Model management
│   └── conversation_controller.py
├── service/            # Business logic services
│   ├── chat_service.py # Main chat orchestration service
│   └── language_model_service.py # LLM provider management
├── core/               # Core utilities
│   └── language_model/llm_provider.py # Unified LLM provider interface
├── model/              # Database models
├── schema/             # Pydantic schemas for API
├── common/config/      # Configuration management
└── main.py             # FastAPI application entry point
```

### Multi-Agent Graph Flow

The main TCM graph (`tcm_builder.py:build_tcm_graph()`) implements the following flow:

```
START
  → middleware_before (guardrails, PII, logging)
  → [conditional: blocked?]
      ├── YES → middleware_after → END
      └── NO → analyze_and_route_query (intent recognition)
          → [route by intent]
              ├── wellness_subgraph_node
              ├── handle_diagnose_query
              ├── handle_herb_query
              ├── handle_prescription_query
              └── respond_to_general_query
          → middleware_after → END
```

### Diagnosis Subgraph Architecture

Located at `agent/components/diagnose/`, this is the most complex component implementing multi-turn TCM diagnosis:

1. **Information Collection Loop** (`collect_info` → `analyze_and_follow_up`)
   - Collects symptoms based on "TCM Ten Questions" (十问歌)
   - Supports multi-modal input (tongue images, lab reports)
   - Multi-round follow-up questions until sufficient info gathered

2. **Complexity Assessment** (`assess_complexity`)
   - Scores case complexity (0-10 points) based on:
     - Symptom count, organ systems involved, duration
     - Contradictory symptoms, chronic conditions, tongue abnormalities
   - Routes to appropriate diagnosis strategy:
     - **Simple (0-3)**: LLM direct diagnosis
     - **Moderate (4-6)**: RAG + predefined Cypher queries
     - **Complex (7-10)**: DeepSearch Agent (multi-source retrieval)

3. **Diagnosis Strategies**:
   - `simple_diagnosis/`: Direct LLM reasoning with TCM theory
   - `moderate_diagnosis/`: Knowledge graph + vector search
   - `complex/complex_diagnosis.py`: DeepSearch with task decomposition

See `agent/components/diagnose/ARCHITECTURE.md` for detailed design.

### Middleware Chain

Middleware (`agent/middleware/`) executes before/after model calls with priority-based ordering:

**Priority Levels** (lower number = earlier execution):
- P0: Guardrails (TCM-specific safety, can block requests via `jump_to: "end"`)
- P1: Model/Tool Retry
- P2: Context Management, Summarization
- P3: Cost Control (call limits)
- P4: Logging, PII redaction

**Middleware Interface** (`middleware/base.py`):
```python
class BaseMiddleware:
    def before_model(state, runtime) -> Optional[Dict]  # Pre-model call
    def after_model(state, runtime) -> Optional[Dict]   # Post-model call
    def wrap_tool_call(tool_call, tool_name, state) -> Callable  # Tool wrapping
```

**Key behaviors**:
- `before_model` returns `{"jump_to": "end"}` to block execution
- State updates accumulate across all middlewares
- `after_model` executes in reverse order (stack-like)

---

## Frontend Structure

```
frontend/src/
├── views/              # Page components by role
│   ├── home/           # Landing page
│   ├── public/         # Public user portal and pages
│   ├── professional/   # Professional user portal
│   └── admin/          # Admin portal
├── router/             # React Router configuration
│   └── ProtectedRoute.tsx # Role-based route protection
├── contexts/           # React contexts
│   └── AuthContext.tsx # Authentication state
├── api/                # API client modules
│   ├── modules/        # API endpoint modules (auth, chat, model, conversation)
│   └── types/          # TypeScript types for API responses
├── components/common/  # Shared UI components
└── constants/          # Constants (prompts, models)
```

### User Roles

Three distinct user roles with separate portals:
- **Public** (`UserRole.PUBLIC`): General users, model management access
- **Professional** (`UserRole.PROFESSIONAL`): TCM practitioners
- **Admin** (`UserRole.ADMIN`): System administrators

### Authentication Flow

- Login/register pages: `/login/{role}`, `/register/{role}`
- Protected portals: `/public`, `/professional`, `/admin`
- Model management: `/public/models` (Public only)

---

## Key Configuration

### Environment Variables

Copy `.env.example` to project root and configure:

```env
# LLM Provider
CHAT_SERVICE=DEEPSEEK  # or OLLAMA
DEEPSEEK_API_KEY=your-key
DEEPSEEK_BASE_URL=https://api.deepseek.com

# Database (PostgreSQL)
POSTGRESQL_DATABASE_NAME=
POSTGRESQL_USER_NAME=
POSTGRESQL_PASSWORD=
POSTGRESQL_HOST=
POSTGRESQL_PORT=

# Neo4j (Knowledge Graph)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=
```

### LLM Provider Configuration

Models can be configured via:
1. Environment variables (fallback)
2. Database (`SystemModelProvider`, `UserProviderConfig` tables)
3. Runtime via `llm_config` parameter in chat requests

The system uses a unified `get_langchain_llm()` function in `core/language_model/llm_provider.py` that supports:
- OpenAI-compatible APIs
- DeepSeek
- Ollama (local models)
- Custom base URLs

---

## Streaming Response Protocol

Chat responses use Server-Sent Events (SSE) with JSON messages:

```json
// Thread initialization
{"type": "thread_init", "thread_id": "uuid"}

// Status messages (from NODE_DISPLAY_REGISTRY)
{"type": "意图识别", "content": "正在识别您的意图..."}

// Content chunks (streaming LLM output)
{"type": "content", "content": "根据您的症状..."}

// Thinking process (if enable_thinking=true)
{"type": "thinking", "content": "分析症状..."}

// User follow-up interrupt (for multi-turn diagnosis)
{"type": "interrupt", "question": "请问您怕冷还是怕热?", "action": "ask_symptom", "thread_id": "uuid"}

// Completion
{"type": "done", "query_type": "tcm-diagnose", "steps": ["意图识别", "信息收集", "辨证分析"]}

// Error
{"type": "error", "content": "错误信息"}
```

After `interrupt`, frontend should call `/api/v1/chat/resume` with user's answer to continue.

---

## Important Notes

### Thinking Mode

The system supports "thinking mode" which exposes LLM reasoning to the frontend. Enabled via `llm_config.enable_thinking` or model selection.

**Supported Provider Formats** (`diagnose/utils/thinking_parser.py`):

| Provider | Format | Extraction Path |
|----------|--------|-----------------|
| DeepSeek, Qwen, GLM | Standard | `chunk.additional_kwargs['reasoning_content']` |
| OpenAI gpt-5 | Reasoning type | `chunk.content[i]['summary']` where `type='reasoning'` |
| Claude, Gemini | Thinking type | `chunk.content[i]['thinking']` where `type='thinking'` |

**Streaming Response**:
- Thinking content: `{"type": "thinking", "content": "..."}`
- Regular content: `{"type": "content", "content": "..."}`

### State Management

- Conversation state persisted via LangGraph `checkpointer` (PostgresSaver)
- Thread ID (`thread_id`) tracks multi-turn conversations
- Use `get_tcm_agent_service().get_conversation_history(thread_id)` to retrieve history

### GraphRAG Integration

Located at `backend/graphrag/`, this module provides hybrid knowledge retrieval combining:
- **Knowledge Graph**: Neo4j-based TCM entities (herbs, syndromes, prescriptions)
- **Vector Search**: Semantic similarity for cases and classics
- **Text Retrieval**: Traditional Chinese medicine literature

Used by moderate and complex diagnosis strategies for knowledge-augmented reasoning.

### Adding New Intent Types

1. Create handler in `agent/components/{new_type}/`
2. Add route in `agent/components/router/router.py:route_query()`
3. Add node to graph in `tcm_builder.py:build_tcm_graph()`
4. Add display name to `schema/chat_schema.py:NODE_DISPLAY_REGISTRY`

### Testing

- Backend tests: `pytest backend/tests/` (when available)
- Frontend tests: `npm test` in `frontend/` (when configured)

---

## API Endpoints

### Chat
- `POST /api/v1/chat/generate` - Generate chat response (streaming)
- `POST /api/v1/chat/resume` - Resume after interrupt
- `POST /api/v1/chat/analyze_persona` - Analyze user persona

### Authentication
- `POST /api/v1/account/register` - User registration
- `POST /api/v1/account/login` - User login
- `GET /api/v1/account/info` - Get user info

### Models
- `GET /api/v1/models/providers` - List available providers
- `POST /api/v1/models/config` - Configure model
- `GET /api/v1/models/config` - Get user's model config

See `/docs` (Swagger UI) when server is running for full API documentation.

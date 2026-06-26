# PRD-033 Fork-Delta Manifest (design-lock #6)

Generated 2026-06-26 for the full-agent Docker migration. The image build is **blocked** until every
item below is marked `baked` (into image), `mounted` (rides /opt/data), `rewritten` (host→container
URL), or `disabled` (intentionally off for first cutover).

Upstream merge-base: `2b3a4f0af80f2952760fdeedb9f26f4eac7faff3`

## A. Committed fork code delta vs upstream (rides into image via `COPY . .`)
All committed; baked into the image automatically (working tree is the build context; `.dockerignore`
excludes only `.git`, `.venv`, `node_modules`, build artifacts).

| Area | Files | PRD | Disposition |
|---|---|---|---|
| Memory orchestrator + chronicle | `plugins/memory/mem0/{__init__.py,orchestrator.py,chronicle.py}` | 006-P4/020 | **baked** |
| Cron memory-write split | `agent/agent_init.py`, `agent/turn_context.py`, `cron/scheduler.py`, `run_agent.py` | 022 | **baked** |
| Session-search caps | `tools/session_search_tool.py`, `tools/budget_config.py` | 023 | **baked** |
| ask_claude tool | `tools/claude_review_tool.py`, `toolsets.py`, `hermes_cli/tools_config.py` | 024 | **baked** (host `claude` CLI absent → FR-8 disabled) |
| Autonomy spine | `autonomy/{__init__,audit,budget,killswitch,redact}.py`, `hermes_cli/main.py` | 028 | **baked**; state → `mounted` `/opt/data/autonomy/` |
| Sandbox backend | `tools/environments/sylva_sandbox.py`, `tools/terminal_tool.py`, `tools/environments/local.py` | 026 FR-7 | **baked but inert** — needs docker.sock (forbidden); FR-9 supersedes |
| Capability policy | `tools/capability_policy.py`, `tools/capability_approvals.py`, `tools/registry.py`, `tools/mcp_tool.py`, `agent/tool_executor.py`, `agent/agent_runtime_helpers.py`, `model_tools.py`, `tools/approval.py` | 032 | **baked** (observe mode); FR-10 re-interpret tiers around container boundary |
| Aux reasoning toggle | `agent/auxiliary_client.py`, `agent/conversation_compression.py` | 020 | **baked** |

## B. Uncommitted working-tree edits (ALSO bake via `COPY . .`)
Pre-existing, not part of PRD-033. Dispositioned `baked as-is` (single-user dev box; acceptable per
FR-1 — image is reproducible from the working tree at build time, SHA stamped via `HERMES_GIT_SHA`).
Flagged for the report; NOT committed by this migration.
- `agent/tool_executor.py` (+1) — **baked as-is**
- `tools/session_search_tool.py` (+69) — **baked as-is** (PRD-023-adjacent local refinement)
- `package-lock.json` (±) — **baked as-is**

## C. Host-path / localhost config that breaks in a container → rewrite or mount
| Item | Current (host) | Action | FR |
|---|---|---|---|
| `config.yaml` model.base_url | `http://localhost:8081/v1` | **rewritten** → `http://llama-qwen36-35b:8080/v1` | FR-4 |
| `config.yaml` custom_providers.local-qwen.base_url | `http://localhost:8081/v1` | **rewritten** → same | FR-4 |
| `config.yaml` auxiliary.vision.base_url | `http://localhost:8081/v1` | **rewritten** → same | FR-4 |
| `config.yaml` auxiliary.compression.base_url | `http://localhost:8081/v1` | **rewritten** → same | FR-4 |
| `mem0.json` vector_store url | `http://localhost:6333` | **rewritten** → `http://qdrant:6333` | FR-4 |
| `mem0.json` embedder openai_base_url | `http://localhost:8085` | **rewritten** → `http://tei-bge-m3:80` | FR-4 |
| `mem0.json` llm openai_base_url | `http://localhost:8082/v1` | **rewritten** → `http://llama-qwen3-4b:8080/v1` | FR-4 |
| `mem0.json` history_db_path | `/home/sgabel/.hermes/mem0_history.db` | **rewritten** → `/opt/data/mem0_history.db` | FR-4 |
| `.env` QDRANT_URL | localhost:6333 | **rewritten** → `http://qdrant:6333` | FR-4 |
| `.env` TEI_BASE_URL | localhost:8085 | **rewritten** → `http://tei-bge-m3:80` | FR-4 |
| `.env` MEM0_LLM_BASE_URL | localhost:8082/v1 | **rewritten** → `http://llama-qwen3-4b:8080/v1` | FR-4 |
| `.env` API_SERVER_HOST | (unset) | **rewritten** → `0.0.0.0` (host exposure loopback-only via publish) | FR-4 |
| context pin | `context_length: 64000` | **mounted** (rides config.yaml) — verify resolves 64000 | FR-2 |
| `handoff` quick-command → `/home/sgabel/hermes/scripts/session-handoff.py` | host path | **disabled** in-container (host path absent) | FR-8 |
| `ask_claude` / Codex (host CLI + OAuth) | host binaries | **disabled** in-container (advisory aids) | FR-8 |
| identity/memory data (`state.db`, sessions, MEMORY.md/USER.md, skills, `auth.json`, `.env`) | `~/.hermes/*` | **mounted** `~/.hermes:/opt/data` | FR-2 |

## D. Voice
- `kokoro-llama-bridge.service` (host, :8091, upstream `127.0.0.1:8081`) → **containerized** on
  `hermes-agent-deps`, upstream rewritten to `http://llama-qwen36-35b:8080` (FR-7).
- `hermes-sylva-voice.service` → **stays host-side** this migration; reaches API via loopback publish
  `127.0.0.1:8642` (FR-7).

## E. Credential hygiene (FR-13)
- `mem0.json` is currently `-rw-rw-r--` → **tighten to 600** before cutover. (`.env`, `auth.json`,
  `config.yaml` already 600.)

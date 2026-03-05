# QVest Project Instructions

Purpose
- Proof-of-concept recommender for a school district library system.
- Runs a FastAPI backend with an agent engine and serves a demo UI.

Project map
- Backend: `qvest/backend` (FastAPI app is in `qvest/backend/app.py`)
- Agent engine: `qvest/backend/agents`
- Tool router + tools: `qvest/backend/tools`
- Data loading and scoring: `qvest/backend/data_loader.py`, `qvest/backend/recommender.py`, `qvest/backend/scoring.py`
- Frontend: `qvest/frontend` (static HTML/JS/CSS, entry at `qvest/frontend/index.html`)
- Data: `qvest/data`
- Assets: `qvest/assets`
- Presentation material: `qvest/presentation.md`, `qvest/index.html`
- Python project metadata: `qvest/pyproject.toml`

Working rules
- Keep edits inside `qvest/` unless the user asks otherwise.
- If changes affect both backend and frontend, call it out explicitly.
- Prefer updating existing modules over creating new ones unless needed.

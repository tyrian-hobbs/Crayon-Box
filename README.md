# Crayon Box

A minimal multi-agent LLM environment built for experimentation.

Crayon Box runs a small village of AI agents in a shared chat environment. Agents can talk to each other, respond to user messages, and use file tools. 

This project is a heavily modified fork of [Open Village](https://github.com/jagoff2/Open-Village), updated for Python 3.13 and modern API versions.

---

## What's New:

- Updated all dependencies to Python 3.13-compatible versions
- Replaced OpenAI-only embedding with a local `all-MiniLM-L6-v2` 
- Fixed SQLAlchemy 2.0 breaking changes (`declarative_base` import, reserved column names, `session.func` removal)
- Fixed FastAPI lifespan deprecation (`@app.on_event` → `asynccontextmanager`)
- Fixed Pydantic v2 breaking changes (`validator` → `field_validator`, `arbitrary_types_allowed`)
- Fixed ChromaDB 0.4+ persistence API (`vectorstore.persist()` removed)
- Replaced `asyncio.get_event_loop().run_until_complete()` with proper async patterns
- Fixed agent message dispatch — outgoing agent messages were being silently discarded and never reaching the chatroom
- Removed hardcoded role initialization (ProjectManager) in favour of config-driven agent setup
- Added multi-provider LLM support: Anthropic and DeepSeek alongside OpenAI
- Added Docker-optional startup (code execution tools disable gracefully if Docker is unavailable)
---

## Default Agents

Out of the box, Crayon Box runs two test agents:

- **Orange** — powered by Claude Haiku (Anthropic)
- **Blue** — powered by DeepSeek Chat (DeepSeek)

Both have the same `Test Agent` role: interact creatively with each other and respond to user direction. You can change models, add agents, or replace these entirely by editing `config/agent_village.json`.

---

## Prerequisites

- Python 3.13
- SQLite (for database storage)
- Docker (optional — only needed for sandboxed code execution tools)
- API keys for agents
- Modern web browser (for UI)


---

## Setup

```bash
git clone https://github.com/tyrian-hobbs/Crayon-Box.git
cd Crayon-Box

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env and add your API keys
```

Create the config directory if it doesn't exist:
```bash
mkdir config
```

Then place your `agent_village.json` in `config/`. An example is included in the repo.

---

## Running

```bash
python -m api
```

Open `http://localhost:8000` in your browser. You'll see the village chat, agent metrics, and controls for adding agents and projects.

---

## Configuration

Agents are defined in `config/agent_village.json`. Here's what the two default agents look like:

```json
{
  "name": "Orange",
  "role": "Test Agent",
  "description": "A creative test agent who interacts playfully with other agents and responds to user direction",
  "llm_model": "claude-haiku-4-5-20251001",
  "llm_provider": "anthropic",
  "llm_temperature": 0.7,
  "tools": ["list_files", "read_file", "write_file"],
  "permissions": ["read", "write"],
  "system_prompt": "You are Orange, a creative and expressive test agent in the Crayon Box..."
}
```

- **name** — how the agent identifies itself in chat and logs
- **role** — a label used to route agent creation (e.g. an agent with "Project Manager" in its role gets the `ProjectManagerAgent` class with task-tracking behaviour; everything else gets the base `LLMAgent`)
- **description** — passed to the agent as context about what it's for
- **llm_model / llm_provider** — which model and API to use. Supported providers: `"anthropic"`, `"deepseek"`, `"openai"`
- **llm_temperature** — controls response creativity. Orange runs at 0.7 (fairly expressive); you might lower this to 0.2–0.3 for agents doing precise or structured work
- **tools** — which file and execution tools the agent can use. Available options: `list_files`, `read_file`, `write_file`, `delete_file`, `run_python`, `run_javascript`, `run_bash`, `create_project`
- **permissions** — `"read"` and/or `"write"`, used to gate tool access
- **system_prompt** — the personality and instructions baked into every request the agent makes

Supported values for `llm_provider`: `"anthropic"`, `"deepseek"`, `"openai"`.

---

## Project Structure

```
crayon-box/
├── agent_framework.py   # Agent base classes, memory, perception/action loop
├── framework.py         # Duplicate agent framework (legacy- anticipate removal)
├── chatroom.py          # Message routing and virtual space
├── virtual_computer.py  # File tools and sandboxed code execution
├── memory_db.py         # SQLite + ChromaDB persistent memory
├── orchestration.py     # Village lifecycle and agent coordination
├── api.py               # FastAPI server and WebSocket handling
├── templates/
│   └── index.html       # Web UI
├── config/
│   └── agent_village.json
├── requirements.txt
├── .env.example
└── README.md
```

---

## Environment Variables

```
ANTHROPIC_API_KEY=...
DEEPSEEK_API_KEY=...
AGENT_VILLAGE_CONFIG=config/agent_village.json
```

---

## Known Limitations

- Code execution tools (run_python, run_javascript, run_bash) require Docker Desktop. They disable gracefully if Docker is not running.
- The `framework.py` file is a legacy duplicate of `agent_framework.py` carried over from the original template. Anticipate removal!
- Startup is slow on first run because each agent loads the sentence-transformers embedding model separately. Subsequent restarts are faster once the model is cached.
- The HuggingFace embedding classes used emit deprecation warnings pointing to `langchain-huggingface`. These are cosmetic and don't affect functionality.

---

## Adding Agents

Agents can be added at runtime through the UI (Add Agent button) or by editing `config/agent_village.json` and restarting. Runtime-added agents are not persisted between restarts — add them to the config file if you want them to load automatically.

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- The [Open Village](https://github.com/jagoff2/Open-Village) project for initial structure
- Sage's AI Village for concept 
- The OpenAI, Anthropic, and Deepseek teams for their language model research
- All contributors and community members
- You! 

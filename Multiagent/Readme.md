# CrewAI + LangGraph Multi-Agent System (Groq-powered)

A minimal two-agent pipeline where **LangGraph** orchestrates the workflow and **CrewAI** defines the agents, with **Groq** serving as the LLM backend for fast inference.

## How it works

```
START -> research_node (Researcher agent) -> write_node (Writer agent) -> END
```

- **LangGraph** owns the control flow and shared state (`topic`, `research`, `article`), moving data from one step to the next.
- **CrewAI** owns the agents themselves. Each LangGraph node spins up a one-task `Crew` and runs it.
- **Groq** (`llama-3.3-70b-versatile`) is the model both agents call under the hood, via CrewAI's native `LLM` class.

| Agent | Role | Input | Output |
|---|---|---|---|
| Researcher | Senior Research Analyst | `topic` | 5–8 bullet-point facts |
| Writer | Technical Content Writer | `topic` + research notes | Short polished article |

## Project files

| File | Purpose |
|---|---|
| `multi_agent_system.py` | The full pipeline: agents, tasks, graph, and CLI runner |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

## Setup

1. **Install dependencies** (Python 3.10+ recommended):
   ```bash
   pip install -r requirements.txt
   ```

2. **Get a Groq API key** — free at [console.groq.com/keys](https://console.groq.com/keys).

3. **Create a `.env` file** in the same folder as the script:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

## Usage

Run with a custom topic:
```bash
python multi_agent_system.py "the future of remote work"
```

Or with no argument (uses a default topic):
```bash
python multi_agent_system.py
```

The script prints the research notes, followed by the final article.

## Customizing

- **Change the model**: edit the `LLM(model="groq/...")` line. See available models at [console.groq.com/docs/models](https://console.groq.com/docs/models).
- **Add another agent**: define a new `Agent`, write a node function that builds a `Task` + single-agent `Crew` and returns a state update, then register it with `workflow.add_node(...)` and wire it in with `workflow.add_edge(...)`.
- **Add branching logic** (e.g. a reviewer that can send the article back to the writer): replace a plain `add_edge` with `workflow.add_conditional_edges(...)`, routing based on a field in the state.
- **Swap Groq for another provider**: CrewAI's `LLM` class supports OpenAI, Anthropic, Gemini, and others — just change the `model` string (e.g. `"openai/gpt-4o"`) and the matching API key env var.

## Troubleshooting

- **`ImportError` on `crewai.LLM`**: your installed CrewAI version is too old. Run `pip install -U crewai`.
- **`GROQ_API_KEY is not set` on launch**: confirm `.env` is in the same directory as the script, or export the variable directly: `export GROQ_API_KEY=...`.
- **Rate limit / 429 errors from Groq**: the free tier has token-per-minute caps; wait a moment and retry, or switch to a smaller model like `llama-3.1-8b-instant`.
- **Garbled or empty agent output**: increase `temperature` for more creative writing, or tighten the `expected_output` field on the relevant `Task` for stricter formatting.

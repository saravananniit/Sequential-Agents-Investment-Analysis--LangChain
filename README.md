# Sequential Agents – Investment Analysis (LangChain)

Port of the Langflow flow **"Sequential Tasks Agents"**:

```
Question ─► Researcher Agent ─► Finance Agent ─► Analysis & Editor Agent ─► Report
            (Tavily search)     (Yahoo! Finance)   (Calculator)
```

All API keys and settings are typed into `index.html`; there is no `.env`.

## 1. Prerequisites
- Python 3.10+
- An Azure OpenAI deployment (endpoint, deployment name, key) **or** an OpenAI key
- A Tavily API key (https://tavily.com)

## 2. Setup
```bash
cd sequential_agents_app
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Run
```bash
uvicorn app:app --reload --port 8000
```
Open http://localhost:8000

## 4. Prove the output
1. In **Configuration**, pick the provider and fill the endpoint / deployment /
   API version / keys (same values you used in Langflow).
2. Click **Test connection** – both LLM and Tavily should show ✓.
3. Keep the default question (or ask your own) and click **Run analysis**.
4. Watch the three step cards: each shows `working…`, then `✓ tools: …`
   (researcher → `tavily_search`, finance → `yahoo_finance`, editor → `calculator`).
5. Expand "Researcher output" / "Finance output" to see the hand-offs, and read
   the final report (recommendation, tables, price targets, images).
6. Compare with the Langflow Playground output for the same question.

## Langflow → LangChain mapping
| Langflow node | Here |
|---|---|
| Chat Input | `question` field in the UI |
| Azure OpenAI ×3 | `AzureChatOpenAI` (or `ChatOpenAI`), one model shared by the 3 agents |
| Researcher Agent + Tavily | `create_agent(... tools=[tavily_search, current_date])` |
| Finance Agent + Yahoo! Finance | `create_agent(... tools=[yahoo_finance, current_date])` |
| Analysis & Editor + Calculator | `create_agent(... tools=[calculator, current_date])` |
| Prompt nodes | `RESEARCHER_PROMPT`, `FINANCE_PROMPT`, `EDITOR_PROMPT` in `agents.py` |
| Chat Output | markdown report rendered in the UI |

## Files
- `agents.py` – prompts, tools, model factory, `run_pipeline()`
- `app.py` – FastAPI: `/api/check`, `/api/run` (streams NDJSON), serves the UI
- `static/index.html` – config form + progress + report

## Notes
- Keys travel in the request body to your own backend only; the optional
  "Remember settings" box stores them in this browser's localStorage – leave it
  off on shared machines. Use HTTPS if you deploy this anywhere.
- The Langflow researcher prompt contained a placeholder line
  (`[Previous content remains the same…]`); `RESEARCHER_PROMPT` is a cleaned
  version of it.
- Each stage builds its own agent, so Finance sees only the researcher's output
  (as in your flow) and the Editor sees both outputs via its system prompt.

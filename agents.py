"""Sequential multi-agent pipeline (LangChain) - port of the Langflow
"Sequential Tasks Agents" flow.

    Question -> Researcher (Tavily) -> Finance (Yahoo Finance)
             -> Analysis & Editor (Calculator) -> Final report

All credentials come from the request (the UI); nothing is read from .env.
"""
from __future__ import annotations

import ast
import json
import operator
from datetime import datetime
from typing import Callable, Iterator

import httpx
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI, ChatOpenAI

# --------------------------------------------------------------------------
# Prompts (taken from the three Prompt nodes of the Langflow flow)
# --------------------------------------------------------------------------
RESEARCHER_PROMPT = """# Expert Research Agent Protocol

You are an expert market researcher. Use the tavily_search tool (several times,
with different queries) to gather current information about the company in the
user's question. Always request images (the tool already includes them).

## Image and Visual Data Handling
1. Collect relevant stock charts, product images, company facilities, key
   executive photos, recent event images and market share visualizations.
2. Save image URLs only from reliable sources; prefer recent, high-quality images.
3. For every image include the full URL, a clear description, source, date and
   why it is relevant, in this format:
   ![Image Description](image_url)
   - Source: [Source Name]
   - Date: [Image Date]
   - Context: [Brief explanation of image relevance]

## Output Structure
### Company Overview
### Recent Developments   (latest news with dates)
### Market Context        (industry trends, competitive position)
### Visual Insights       (reference relevant images)
### Key Risk Factors
### Sources

Remember to:
- Use Markdown formatting for clear structure
- Include dates for all time-sensitive information
- Quote significant statistics and statements
- Highlight conflicting information or viewpoints
- Pass all gathered data on for detailed financial analysis
"""

FINANCE_PROMPT = """# Financial Analysis Expert Protocol

You are an elite financial analyst with access to a Yahoo! Finance tool
(yahoo_finance). Perform a comprehensive financial analysis based on the
research provided in the user message and the data available from Yahoo! Finance.

## CRITICAL: Stock Symbol Usage
- Always use correct ticker symbols in UPPERCASE (AAPL, MSFT, NVDA, GOOGL, TSLA).
- Never use company names, lowercase, a leading $ or exchange extensions.

## Data Collection Strategy
1. Verify the symbol first with get_info, cross-check with get_fast_info. If the
   symbol is invalid, report the error immediately.
2. Core company analysis: get_info, get_fast_info, get_earnings, get_calendar
3. Financial statements: get_income_stmt, get_balance_sheet, get_cashflow
4. Market intelligence: get_recommendations, get_recommendations_summary,
   get_upgrades_downgrades, get_news (set num_news)
5. Ownership: get_institutional_holders, get_major_holders,
   get_mutualfund_holders, get_insider_purchases, get_insider_transactions,
   get_insider_roster_holders
6. History: get_actions, get_dividends, get_splits, get_capital_gains,
   get_sec_filings, get_sustainability

## Analysis Framework
Profitability (revenue trends, margins, efficiency, returns); Financial health
(liquidity, debt, working capital, cash-flow quality); Growth (historical,
projected, opportunity); Risk (financial, market position, operational,
competitive).

## Output Structure
### Symbol Information
### Financial Overview
### Profitability Analysis
### Balance Sheet Review
### Cash Flow Assessment
### Market Sentiment
### Growth Analysis
### Risk Factors

Remember to: verify the symbol first, use exact numbers from the data, compare
with industry standards, highlight trends, flag data anomalies, provide metric
context. Include any invalid-symbol errors or data gaps in your report.
"""

EDITOR_PROMPT = """# Investment Analysis & Editorial Protocol

You are an elite financial analyst and editorial expert responsible for the
final investment analysis report. Synthesize the research and financial data
into a visually appealing, data-rich report using proper markdown.
Use the calculator tool for arithmetic (percent changes, targets, ratios), but
keep it to at most ~8 calls, then write the final report. Do not call tools
after the report is complete.

## Input Processing
1. Research Agent Input (market research, news, images, sentiment):
{research_agent_output}

2. Finance Agent Input (financial metrics, ratings, growth, risks):
{finance_agent_output}

## Formatting Requirements
- Single # for the main title, increment for subsections
- Place images right after the relevant section: ![Alt](url) with *italic* caption
- Standard markdown tables, numbers right-aligned, header separators
- Bold (**) key metrics, show percentage changes and trends (↑/↓)

## Report Structure
# Investment Analysis Report: [Company Name] ($SYMBOL)
*Generated: [Date] | Type: Comprehensive Evaluation*

[Executive Summary - 3 paragraphs max]

## Quick Take
- **Recommendation**: [BUY/HOLD/SELL]
- **Target Price**: $XXX
- **Risk Level**: [LOW/MEDIUM/HIGH]
- **Investment Horizon**: [SHORT/MEDIUM/LONG]-term

## Market Analysis   (most relevant market image, Source + context)
### Industry Position
## Financial Health  (table: Metric | Value | YoY Change | Industry Avg)
### Key Performance Indicators (Revenue Growth, Profit Margin, ROE)
## Growth Drivers   (short-term catalysts, long-term opportunities, pipeline)
## Risk Assessment  (table: Risk Factor | Severity | Probability | Impact)
## Technical Analysis (chart image if available)
## Investment Strategy (long 18+ months, medium 6-18, short 0-6)
## Price Targets    (Bear / Base / Bull, with % vs current)
## Monitoring Checklist
## Visual Evidence
*Disclaimer: This analysis is for informational purposes only. Always conduct
your own research before making investment decisions.*

Remember: never use triple backticks, include all images with proper markdown,
keep formatting consistent, give specific actionable insights, cross-validate
all data points.
"""

# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval(node.operand)
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    raise ValueError("Unsupported expression")


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression, e.g. '4*4*(33/22)+12-20'.
    Supports + - * / ** and parentheses."""
    try:
        result = _eval(ast.parse(expression, mode="eval").body)
        return f"{result:.6f}".rstrip("0").rstrip(".")
    except ZeroDivisionError:
        return "Error: Division by zero"
    except Exception as exc:  # noqa: BLE001
        return f"Invalid expression: {exc}"


@tool
def current_date() -> str:
    """Return the current date (YYYY-MM-DD)."""
    return datetime.now().strftime("%Y-%m-%d")


YF_METHODS = [
    "get_info", "get_news", "get_actions", "get_analysis", "get_balance_sheet",
    "get_calendar", "get_cashflow", "get_institutional_holders",
    "get_recommendations", "get_sustainability", "get_major_holders",
    "get_mutualfund_holders", "get_insider_purchases",
    "get_insider_transactions", "get_insider_roster_holders", "get_dividends",
    "get_capital_gains", "get_splits", "get_shares", "get_fast_info",
    "get_sec_filings", "get_recommendations_summary", "get_upgrades_downgrades",
    "get_earnings", "get_income_stmt",
]


@tool(description=(
    "Fetch Yahoo! Finance data for a stock ticker (UPPERCASE, e.g. TSLA). "
    "`method` must be one of: " + ", ".join(YF_METHODS) + ". "
    "`num_news` only applies to get_news."))
def yahoo_finance(symbol: str, method: str = "get_info", num_news: int = 5) -> str:
    import pprint
    import yfinance as yf

    if method not in YF_METHODS:
        return f"Error: unknown method '{method}'. Valid: {', '.join(YF_METHODS)}"
    try:
        ticker = yf.Ticker(symbol.strip().upper().lstrip("$"))
        if method == "get_info":
            data = ticker.info
        elif method == "get_news":
            data = ticker.news[:num_news]
        elif method == "get_fast_info":
            data = dict(ticker.fast_info)
        else:
            data = getattr(ticker, method)()
        text = pprint.pformat(data)
        return text[:12000]  # keep tool output within a sane context size
    except Exception as exc:  # noqa: BLE001
        return f"Error retrieving data: {exc}"


def make_tavily_tool(api_key: str):
    @tool
    def tavily_search(query: str) -> str:
        """Search the web with Tavily (advanced depth, includes a short answer
        and related image URLs). Use specific queries."""
        payload = {
            "api_key": api_key, "query": query, "search_depth": "advanced",
            "topic": "general", "max_results": 5, "include_images": True,
            "include_answer": True, "chunks_per_source": 3,
        }
        try:
            r = httpx.post("https://api.tavily.com/search", json=payload, timeout=90)
            r.raise_for_status()
            data = r.json()
        except httpx.HTTPStatusError as exc:
            return f"HTTP error: {exc.response.status_code} - {exc.response.text[:300]}"
        except Exception as exc:  # noqa: BLE001
            return f"Tavily request failed: {exc}"
        out = []
        if data.get("answer"):
            out.append(f"ANSWER: {data['answer']}")
        for res in data.get("results", []):
            out.append(f"- {res.get('title')} ({res.get('url')})\n  {res.get('content', '')[:900]}")
        if data.get("images"):
            imgs = [i if isinstance(i, str) else i.get("url", "") for i in data["images"]]
            out.append("IMAGES:\n" + "\n".join(imgs[:8]))
        return "\n".join(out) or "No results."

    return tavily_search


# --------------------------------------------------------------------------
# Model + pipeline
# --------------------------------------------------------------------------
def build_llm(cfg: dict):
    provider = cfg.get("provider", "azure")
    temp = float(cfg.get("temperature", 0.7))
    if provider == "azure":
        return AzureChatOpenAI(
            azure_endpoint=cfg["azure_endpoint"],
            azure_deployment=cfg["azure_deployment"],
            api_version=cfg.get("api_version") or "2025-02-01-preview",
            api_key=cfg["api_key"],
            temperature=temp,
        )
    return ChatOpenAI(
        model=cfg["model"],
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url") or None,
        temperature=temp,
    )


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


# Each tool call costs ~2 graph steps; never go below this, whatever the UI says.
MIN_RECURSION = 40


def _run_agent(llm, tools, system_prompt: str, user_input: str, max_iter: int) -> tuple[str, list[str]]:
    agent = create_agent(model=llm, tools=tools, system_prompt=system_prompt)
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config={"recursion_limit": max(MIN_RECURSION, max_iter * 2 + 1)},
    )
    msgs = result["messages"]
    used = [tc["name"] for m in msgs for tc in (getattr(m, "tool_calls", None) or [])]
    return _text(msgs[-1].content), used


def run_pipeline(cfg: dict, question: str, llm_factory: Callable = build_llm) -> Iterator[dict]:
    """Yield progress events: stage_start / stage_done / final / error."""
    max_iter = int(cfg.get("max_iterations", 15))
    try:
        llm = llm_factory(cfg)
        tavily = make_tavily_tool(cfg["tavily_api_key"])

        yield {"type": "stage_start", "stage": 1, "name": "Researcher Agent"}
        research, used1 = _run_agent(llm, [tavily, current_date], RESEARCHER_PROMPT, question, max_iter)
        yield {"type": "stage_done", "stage": 1, "tools": used1, "output": research}

        yield {"type": "stage_start", "stage": 2, "name": "Finance Agent"}
        finance, used2 = _run_agent(llm, [yahoo_finance, current_date], FINANCE_PROMPT, research, max_iter)
        yield {"type": "stage_done", "stage": 2, "tools": used2, "output": finance}

        yield {"type": "stage_start", "stage": 3, "name": "Analysis & Editor Agent"}
        editor_prompt = (EDITOR_PROMPT
                         .replace("{research_agent_output}", research)
                         .replace("{finance_agent_output}", finance))
        report, used3 = _run_agent(llm, [calculator, current_date], editor_prompt,
                                   "Start the analysis", max_iter)
        yield {"type": "stage_done", "stage": 3, "tools": used3, "output": report}
        yield {"type": "final", "report": report}
    except Exception as exc:  # noqa: BLE001
        yield {"type": "error", "message": f"{type(exc).__name__}: {exc}"}


def check_config(cfg: dict) -> dict:
    """Ping the LLM and Tavily so the UI can validate keys before a long run."""
    out = {"llm": {"ok": False, "detail": ""}, "tavily": {"ok": False, "detail": ""}}
    try:
        reply = build_llm(cfg).invoke("Reply with the single word: pong")
        out["llm"] = {"ok": True, "detail": _text(reply.content)[:80]}
    except Exception as exc:  # noqa: BLE001
        out["llm"]["detail"] = f"{type(exc).__name__}: {exc}"[:300]
    try:
        r = httpx.post("https://api.tavily.com/search", timeout=30,
                       json={"api_key": cfg.get("tavily_api_key", ""), "query": "test", "max_results": 1})
        out["tavily"] = {"ok": r.status_code == 200, "detail": f"HTTP {r.status_code}"}
    except Exception as exc:  # noqa: BLE001
        out["tavily"]["detail"] = f"{type(exc).__name__}: {exc}"[:300]
    return out

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, EmailStr
from langgraph.graph import StateGraph, END, START
from llama_stack_client import LlamaStackClient
from typing import Any, Optional, Union
from typing_extensions import TypedDict

import os
import re
import json
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
BASE_URL = os.getenv("LLAMA_STACK_BASE_URL", "http://localhost:8321")
TAVILY_SEARCH_API_KEY = os.getenv("TAVILY_SEARCH_API_KEY", "")
FASTAPI_HOST = os.getenv("FASTAPI_HOST", "0.0.0.0")
FASTAPI_PORT = int(os.getenv("FASTAPI_PORT", "8000"))

# Comma-separated list of allowed model identifiers. Empty means allow all.
_allowed_raw = os.getenv("ALLOWED_MODELS", "")
ALLOWED_MODELS = [m.strip() for m in _allowed_raw.split(",") if m.strip()] if _allowed_raw else []

# Client-side input shields (checked in LangGraph input_safety_node)
INPUT_SHIELDS = ["content_safety", "prompt_injection"]

# Server-side guardrails (applied via extra_body, checks both input and output)
GUARDRAILS = ["hap"]

# Llama Guard safety categories to IGNORE for business queries.
# S5 (Defamation), S7 (Privacy), S8 (Intellectual Property) produce false positives
# when customer names match famous people or contain emails/personal data.
IGNORED_SAFETY_CATEGORIES = {"S5", "S7", "S8"}

INSTRUCTIONS = (
    "You are a FantaCo company assistant. "
    "You can look up customer profiles, orders, and invoices. "
    "When given an email, first use search_customers to find the customer, "
    "then use the customer_id to call get_customer, fetch_order_history, "
    "or fetch_invoice_history as needed. "
    "For general questions, use web_search to find up-to-date information. "
    "Return structured, clear data."
)

logger.info("Configuration loaded:")
logger.info("  Base URL: %s", BASE_URL)
logger.info("  Tavily API Key: %s", "configured" if TAVILY_SEARCH_API_KEY else "NOT SET")
logger.info("  Input Shields: %s", INPUT_SHIELDS)
logger.info("  Guardrails: %s", GUARDRAILS)
logger.info("  FastAPI Host: %s", FASTAPI_HOST)
logger.info("  FastAPI Port: %s", FASTAPI_PORT)
logger.info("  Allowed Models: %s", ALLOWED_MODELS if ALLOWED_MODELS else "ALL")

# Initialize Llama Stack client
client = LlamaStackClient(
    base_url=BASE_URL,
    timeout=120.0,
    provider_data={"tavily_search_api_key": TAVILY_SEARCH_API_KEY},
)

# Resolve default model: first from ALLOWED_MODELS, otherwise first from Llama Stack
if ALLOWED_MODELS:
    DEFAULT_MODEL = ALLOWED_MODELS[0]
else:
    _models = client.models.list()
    DEFAULT_MODEL = _models[0].id if _models else ""
logger.info("  Default Model: %s", DEFAULT_MODEL)

# Auto-discover MCP tools from registered toolgroups
toolgroups = client.toolgroups.list()
mcp_tools = []
for tg in toolgroups:
    if tg.mcp_endpoint:
        label = tg.identifier.replace("mcp::", "")
        logger.info("  MCP toolgroup: %s -> %s", tg.identifier, tg.mcp_endpoint.uri)
        mcp_tools.append({
            "type": "mcp",
            "server_label": label,
            "server_url": tg.mcp_endpoint.uri,
        })

# Combine MCP tools + web_search
TOOLS = mcp_tools + [{"type": "web_search"}]
logger.info("  Tools (%d): %s", len(TOOLS), [t.get("server_label", t["type"]) for t in TOOLS])


# --- Reasoning execution ---

def run_reasoning(user_message: str, model: str | None = None) -> dict:
    """Execute reasoning via responses API.
    Server handles: guardrails (hap) + LLM + MCP tools + web search."""
    model = model or DEFAULT_MODEL

    stream = client.responses.create(
        model=model,
        input=user_message,
        instructions=INSTRUCTIONS,
        tools=TOOLS,
        stream=True,
        extra_body={"guardrails": GUARDRAILS},
    )

    final_text = ""
    tool_calls = []
    blocked = False
    shield_msg = ""

    for event in stream:
        event_type = getattr(event, "type", None)

        if event_type == "response.completed":
            response = event.response
            for item in response.output:
                if getattr(item, "type", None) == "message":
                    for part in getattr(item, "content", []):
                        if getattr(part, "type", None) == "output_text":
                            final_text = part.text
                        elif getattr(part, "type", None) == "refusal":
                            blocked = True
                            shield_msg = part.refusal

        elif event_type == "response.output_item.added":
            item = getattr(event, "item", None)
            if item:
                item_type = getattr(item, "type", None)
                if item_type == "mcp_call":
                    name = getattr(item, "name", "")
                    tool_calls.append({"tool_name": name, "type": "mcp"})
                    logger.info("  [MCP] %s", name)
                elif item_type == "web_search_call":
                    tool_calls.append({"tool_name": "web_search", "type": "web_search"})
                    logger.info("  [WebSearch]")

        elif event_type == "response.mcp_call.completed":
            logger.info("    mcp call completed")

        elif event_type == "response.web_search_call.completed":
            logger.info("    web search completed")

    return {
        "text": final_text,
        "tool_calls": tool_calls,
        "blocked": blocked,
        "shield_message": shield_msg if blocked else "",
        "error": None,
    }


def run_reasoning_streaming(user_message: str, model: str | None = None):
    """Execute reasoning via responses API with streaming. Yields SSE events."""
    model = model or DEFAULT_MODEL

    stream = client.responses.create(
        model=model,
        input=user_message,
        instructions=INSTRUCTIONS,
        tools=TOOLS,
        stream=True,
        extra_body={"guardrails": GUARDRAILS},
    )

    for event in stream:
        event_type = getattr(event, "type", None)

        # Text streaming
        if event_type == "response.output_text.delta":
            yield {"type": "token", "content": event.delta}

        # Refusal (shield violation)
        elif event_type == "response.refusal.delta":
            yield {"type": "token", "content": event.delta}

        # MCP tool calls
        elif event_type == "response.output_item.added":
            item = getattr(event, "item", None)
            if item:
                item_type = getattr(item, "type", None)
                if item_type == "mcp_call":
                    name = getattr(item, "name", "")
                    yield {"type": "tool_call", "tool": name}
                elif item_type == "web_search_call":
                    yield {"type": "tool_call", "tool": "web_search"}

        # Completed response
        elif event_type == "response.completed":
            response = event.response
            for item in response.output:
                if getattr(item, "type", None) == "message":
                    for part in getattr(item, "content", []):
                        if getattr(part, "type", None) == "refusal":
                            yield {"type": "blocked", "content": part.refusal}
                            yield {"type": "end"}
                            return

    yield {"type": "end"}


# --- LangGraph: State and Nodes ---

class GraphState(TypedDict):
    query: str
    model: str
    input_blocked: bool
    block_message: str
    result: dict[str, Any] | None


def _extract_violated_categories(violation) -> set:
    """Extract Llama Guard category codes (S1-S13) from a SafetyViolation."""
    categories = set()
    metadata = getattr(violation, "metadata", {}) or {}
    logger.info("  [Safety] Violation metadata: %s", metadata)
    for key, val in metadata.items():
        if isinstance(val, str):
            categories.update(re.findall(r"S\d+", val))
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, str):
                    categories.update(re.findall(r"S\d+", item))
    user_msg = getattr(violation, "user_message", "") or ""
    categories.update(re.findall(r"S\d+", user_msg))
    return categories


def _is_violation_ignorable(violation) -> bool:
    """Check if all violated categories are in the ignore list."""
    categories = _extract_violated_categories(violation)
    if not categories:
        return False
    ignored = categories.issubset(IGNORED_SAFETY_CATEGORIES)
    if ignored:
        logger.info("  [Safety] All violated categories %s are ignorable, allowing through", categories)
    return ignored


def input_safety_node(state: GraphState) -> dict:
    """Client-side input safety: content_safety + prompt_injection."""
    query = state["query"]
    logger.info("  [InputSafety] Checking: %s", query[:80])

    for shield_id in INPUT_SHIELDS:
        try:
            result = client.safety.run_shield(
                shield_id=shield_id,
                messages=[{"role": "user", "content": query}],
                params={},
            )
            if result.violation:
                level = getattr(result.violation, "violation_level", "unknown")
                msg = getattr(result.violation, "user_message", "Request blocked by safety policy")

                # info level = content verified/passed, not a real violation
                if level == "info":
                    logger.info("  [InputSafety] %s: passed (info)", shield_id)
                    continue

                logger.warning("  [InputSafety] %s violation (level=%s): %s", shield_id, level, msg)

                # Category-based filtering only applies to content_safety
                if shield_id == "content_safety" and _is_violation_ignorable(result.violation):
                    logger.info("  [InputSafety] Violation ignored (business data false positive)")
                    continue

                logger.warning("  [InputSafety] BLOCKED by %s: %s", shield_id, msg)
                return {"input_blocked": True, "block_message": f"[{shield_id}] {msg}"}
        except Exception as e:
            logger.warning("  [InputSafety] %s error (allowing through): %s", shield_id, e)

    logger.info("  [InputSafety] PASSED")
    return {"input_blocked": False, "block_message": ""}


def reasoning_node(state: GraphState) -> dict:
    """Call the Llama Stack responses API."""
    logger.info("  [Reasoning] Processing: %s (model: %s)", state["query"][:80], state.get("model", DEFAULT_MODEL))
    result = run_reasoning(state["query"], model=state.get("model"))

    status = "BLOCKED" if result.get("blocked") else "ERROR" if result.get("error") else "OK"
    tools = len(result.get("tool_calls", []))
    logger.info("  [Reasoning] %s | tools used: %d", status, tools)

    return {"result": result}


def route_after_safety(state: GraphState) -> str:
    if state.get("input_blocked"):
        return "blocked"
    return "reasoning"


# Build the graph: input_safety -> (route) -> reasoning -> END
builder = StateGraph(GraphState)
builder.add_node("input_safety", input_safety_node)
builder.add_node("reasoning", reasoning_node)
builder.add_edge(START, "input_safety")
builder.add_conditional_edges("input_safety", route_after_safety, {
    "blocked": END,
    "reasoning": "reasoning",
})
builder.add_edge("reasoning", END)
graph = builder.compile()
logger.info("LangGraph compiled: START -> input_safety -> (route) -> reasoning -> END")


# --- FastAPI app ---

app = FastAPI(title="Customer Orders and Invoices API")


# Response models
class Customer(BaseModel):
    customerId: str
    companyName: Optional[str] = None
    contactName: Optional[str] = None
    contactEmail: Optional[str] = None


class Order(BaseModel):
    id: Optional[Union[str, int]] = None
    orderId: Optional[Union[str, int]] = None
    orderNumber: Optional[str] = None
    orderDate: Optional[str] = None
    status: Optional[str] = None
    totalAmount: Optional[Union[str, int, float]] = None
    freight: Optional[Union[str, int, float]] = None


class Invoice(BaseModel):
    id: Optional[Union[str, int]] = None
    invoiceId: Optional[Union[str, int]] = None
    invoiceNumber: Optional[str] = None
    invoiceDate: Optional[str] = None
    status: Optional[str] = None
    totalAmount: Optional[Union[str, int, float]] = None
    amount: Optional[Union[str, int, float]] = None
    customerId: Optional[str] = None
    customerEmail: Optional[str] = None
    contactName: Optional[str] = None


class OrdersResponse(BaseModel):
    customer: Optional[Customer] = None
    orders: list[Order] = []
    total_orders: int = 0


class InvoicesResponse(BaseModel):
    customer: Optional[Customer] = None
    invoices: list[Invoice] = []
    total_invoices: int = 0


def sse_event(data: dict) -> str:
    """Format data as an SSE event"""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


@app.get("/models")
async def list_models():
    """List available models from Llama Stack."""
    try:
        models = client.models.list()
        model_list = []
        for m in models:
            if ALLOWED_MODELS and m.id not in ALLOWED_MODELS:
                continue
            metadata = m.custom_metadata or {}
            model_list.append({
                "id": m.id,
                "owned_by": m.owned_by,
                "model_type": metadata.get("model_type"),
            })
        return {
            "models": model_list,
            "default_model": DEFAULT_MODEL,
        }
    except Exception as e:
        logger.error("Error listing models: %s", str(e))
        return {"models": [], "default_model": DEFAULT_MODEL, "error": str(e)}


@app.get("/")
def read_root():
    return {
        "message": "Customer Orders and Invoices API (Llama Stack Responses API)",
        "endpoints": {
            "models": "/models  (list available models)",
            "find_orders": "/find_orders?email=<customer_email>&model=<model_id>  (SSE stream)",
            "find_invoices": "/find_invoices?email=<customer_email>&model=<model_id>  (SSE stream)",
            "question": "/question?q=<your_question>&model=<model_id>  (SSE stream)",
        },
        "features": {
            "tools": [t.get("server_label", t["type"]) for t in TOOLS],
            "input_shields": INPUT_SHIELDS,
            "guardrails": GUARDRAILS,
        },
    }


@app.get("/find_orders")
async def find_orders(email: EmailStr, model: Optional[str] = None):
    """Find all orders for a customer by email address (SSE stream)"""
    logger.info("=" * 80)
    logger.info("API: Finding orders for: %s (model: %s)", email, model or DEFAULT_MODEL)
    logger.info("=" * 80)

    async def event_generator():
        try:
            yield sse_event({"type": "status", "content": f"Searching orders for {email}..."})

            result = graph.invoke({
                "query": f"Find all orders for {email}",
                "model": model or "",
                "input_blocked": False,
                "block_message": "",
                "result": None,
            })

            if result.get("input_blocked"):
                yield sse_event({"type": "blocked", "content": result.get("block_message", "Blocked")})
                yield sse_event({"type": "end"})
                return

            reasoning_result = result.get("result", {})
            if reasoning_result.get("blocked"):
                yield sse_event({"type": "blocked", "content": reasoning_result.get("shield_message", "Blocked")})
                yield sse_event({"type": "end"})
                return

            if reasoning_result.get("error"):
                yield sse_event({"type": "error", "content": reasoning_result["error"]})
                return

            # Return the LLM's text response
            yield sse_event({
                "type": "result",
                "data": {
                    "text": reasoning_result.get("text", ""),
                    "tool_calls": reasoning_result.get("tool_calls", []),
                },
            })
            yield sse_event({"type": "end"})

        except Exception as e:
            logger.error("Error finding orders: %s", str(e))
            yield sse_event({"type": "error", "content": f"Error finding orders: {str(e)}"})

    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=SSE_HEADERS)


@app.get("/find_invoices")
async def find_invoices(email: EmailStr, model: Optional[str] = None):
    """Find all invoices for a customer by email address (SSE stream)"""
    logger.info("=" * 80)
    logger.info("API: Finding invoices for: %s (model: %s)", email, model or DEFAULT_MODEL)
    logger.info("=" * 80)

    async def event_generator():
        try:
            yield sse_event({"type": "status", "content": f"Searching invoices for {email}..."})

            result = graph.invoke({
                "query": f"Find all invoices for {email}",
                "model": model or "",
                "input_blocked": False,
                "block_message": "",
                "result": None,
            })

            if result.get("input_blocked"):
                yield sse_event({"type": "blocked", "content": result.get("block_message", "Blocked")})
                yield sse_event({"type": "end"})
                return

            reasoning_result = result.get("result", {})
            if reasoning_result.get("blocked"):
                yield sse_event({"type": "blocked", "content": reasoning_result.get("shield_message", "Blocked")})
                yield sse_event({"type": "end"})
                return

            if reasoning_result.get("error"):
                yield sse_event({"type": "error", "content": reasoning_result["error"]})
                return

            yield sse_event({
                "type": "result",
                "data": {
                    "text": reasoning_result.get("text", ""),
                    "tool_calls": reasoning_result.get("tool_calls", []),
                },
            })
            yield sse_event({"type": "end"})

        except Exception as e:
            logger.error("Error finding invoices: %s", str(e))
            yield sse_event({"type": "error", "content": f"Error finding invoices: {str(e)}"})

    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=SSE_HEADERS)


@app.get("/question")
async def ask_question(q: str, model: Optional[str] = None):
    """Answer a natural language question (SSE stream with token-level output + safety shields)"""
    logger.info("=" * 80)
    logger.info("API: Processing question (stream): %s (model: %s)", q, model or DEFAULT_MODEL)
    logger.info("=" * 80)

    async def event_generator():
        try:
            # Client-side input safety check
            logger.info("  [Question] Checking safety for: '%s'", q[:80])
            for shield_id in INPUT_SHIELDS:
                try:
                    safety_result = client.safety.run_shield(
                        shield_id=shield_id,
                        messages=[{"role": "user", "content": q}],
                        params={},
                    )
                    if safety_result.violation:
                        level = getattr(safety_result.violation, "violation_level", "unknown")
                        msg = getattr(safety_result.violation, "user_message", "Blocked by safety policy")

                        if level == "info":
                            logger.info("  [Question] %s: passed (info)", shield_id)
                            continue

                        logger.warning("  [Question] %s violation (level=%s): %s", shield_id, level, msg)

                        if shield_id == "content_safety" and _is_violation_ignorable(safety_result.violation):
                            logger.info("  [Question] Violation ignored (business data false positive)")
                            continue

                        yield sse_event({"type": "blocked", "content": f"[{shield_id}] {msg}"})
                        yield sse_event({"type": "end"})
                        return
                except Exception as e:
                    logger.warning("  [Question] %s check error (allowing through): %s", shield_id, e)

            logger.info("  [Question] PASSED input safety")

            # Stream reasoning response
            has_content = False
            for event in run_reasoning_streaming(q, model=model):
                yield sse_event(event)
                if event.get("type") == "token":
                    has_content = True
                if event.get("type") in ("blocked", "error", "end"):
                    if event.get("type") in ("blocked", "error"):
                        return
                    break

            if not has_content:
                yield sse_event({"type": "token", "content": "No response generated"})
                yield sse_event({"type": "end"})

        except Exception as e:
            logger.error("Streaming error: %s", str(e))
            yield sse_event({"type": "error", "content": str(e)})

    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=SSE_HEADERS)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=FASTAPI_HOST, port=FASTAPI_PORT)

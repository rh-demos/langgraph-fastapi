from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, EmailStr
from langgraph.graph import StateGraph, END, START
from llama_stack_client import LlamaStackClient
from llama_stack_client.types import AgentConfig
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

INPUT_SHIELDS = ["prompt_injection"]
OUTPUT_SHIELDS = ["hap"]

# Llama Guard safety categories to IGNORE for business queries.
# S5 (Defamation), S7 (Privacy), S8 (Intellectual Property) produce false positives
# when customer names match famous people or contain emails/personal data.
IGNORED_SAFETY_CATEGORIES = {"S5", "S7", "S8"}

logger.info("Configuration loaded:")
logger.info("  Base URL: %s", BASE_URL)
logger.info("  Tavily API Key: %s", "configured" if TAVILY_SEARCH_API_KEY else "NOT SET")
logger.info("  Input Shields: %s", INPUT_SHIELDS)
logger.info("  Output Shields: %s", OUTPUT_SHIELDS)
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
    DEFAULT_MODEL = _models[0].identifier if _models else ""
logger.info("  Default Model: %s", DEFAULT_MODEL)

# Create Agent with toolgroups and shields
SAMPLING_PARAMS = {
    "strategy": {"type": "top_p", "temperature": 0.7, "top_p": 0.9},
    "max_tokens": 4096,
}

AGENT_INSTRUCTIONS = (
    "You are a FantaCo company assistant. "
    "You can look up customer profiles, orders, and invoices. "
    "When given an email, first use search_customers to find the customer, "
    "then use the customer_id to call get_customer, fetch_order_history, "
    "or fetch_invoice_history as needed. "
    "For general questions, use web_search to find up-to-date information. "
    "Return structured, clear data."
)

AGENT_TOOLGROUPS = ["mcp::customer", "mcp::finance", "builtin::websearch"]

# Cache: model_id -> agent_id
_agent_cache: dict[str, str] = {}


def get_or_create_agent(model: str | None = None) -> str:
    """Get an existing agent for the given model, or create one."""
    model = model or DEFAULT_MODEL

    if model in _agent_cache:
        return _agent_cache[model]

    config = AgentConfig(
        model=model,
        instructions=AGENT_INSTRUCTIONS,
        toolgroups=AGENT_TOOLGROUPS,
        input_shields=INPUT_SHIELDS,
        output_shields=OUTPUT_SHIELDS,
        sampling_params=SAMPLING_PARAMS,
        max_infer_iters=10,
    )
    resp = client.alpha.agents.create(agent_config=config)
    _agent_cache[model] = resp.agent_id
    logger.info("Agent created for model '%s': %s", model, resp.agent_id)
    return resp.agent_id


# Pre-create the default agent
default_agent_id = get_or_create_agent(DEFAULT_MODEL)
logger.info("Default agent: %s (model: %s)", default_agent_id, DEFAULT_MODEL)
logger.info("  toolgroups: %s", AGENT_TOOLGROUPS)
logger.info("  input_shields: %s", INPUT_SHIELDS)
logger.info("  output_shields: %s", OUTPUT_SHIELDS)


# --- Llama Stack Agent turn execution ---

def run_agent_turn(user_message: str, model: str | None = None) -> dict:
    """Execute a single agent turn. Llama Stack handles: shields + LLM + MCP tools."""
    aid = get_or_create_agent(model)
    session = client.alpha.agents.session.create(aid, session_name="")

    stream = client.alpha.agents.turn.create(
        session.session_id,
        agent_id=aid,
        messages=[{"role": "user", "content": user_message}],
        stream=True,
    )

    final_text = ""
    tool_calls = []
    tool_outputs = []
    blocked = False
    shield_msg = ""

    for chunk in stream:
        if hasattr(chunk, "error") and chunk.error:
            return {"text": "", "tool_calls": [], "tool_outputs": [],
                    "blocked": False, "error": str(chunk.error)}

        if not (chunk.event and chunk.event.payload):
            continue

        payload = chunk.event.payload

        if payload.event_type == "turn_complete":
            content = payload.turn.output_message.content
            if isinstance(content, str):
                final_text = content
            elif isinstance(content, list):
                final_text = "".join(
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in content
                )

        elif payload.event_type == "step_complete":
            step_type = getattr(payload, "step_type", None)

            if step_type == "tool_execution" and hasattr(payload.step_details, "tool_calls"):
                for tc in payload.step_details.tool_calls:
                    tc_info = {
                        "tool_name": str(getattr(tc, "tool_name", getattr(tc, "name", ""))),
                        "arguments": getattr(tc, "arguments", {}),
                    }
                    tool_calls.append(tc_info)
                    logger.info("  [Tool] %s(%s)", tc_info["tool_name"], tc_info["arguments"])

                if hasattr(payload.step_details, "tool_responses"):
                    for tr in payload.step_details.tool_responses:
                        output = getattr(tr, "content", "")
                        tool_outputs.append(output)

            elif step_type == "shield_call":
                violation = getattr(payload.step_details, "violation", None)
                if violation:
                    blocked = True
                    shield_msg = getattr(violation, "user_message", "Blocked by safety policy")
                    logger.warning("  [Shield] BLOCKED: %s", shield_msg)
                else:
                    logger.info("  [Shield] PASSED (no violation)")

    return {
        "text": final_text,
        "tool_calls": tool_calls,
        "tool_outputs": tool_outputs,
        "blocked": blocked,
        "shield_message": shield_msg if blocked else "",
        "error": None,
    }


def run_agent_turn_streaming(user_message: str, model: str | None = None):
    """Execute a single agent turn with streaming. Yields SSE events."""
    aid = get_or_create_agent(model)
    session = client.alpha.agents.session.create(aid, session_name="")

    stream = client.alpha.agents.turn.create(
        session.session_id,
        agent_id=aid,
        messages=[{"role": "user", "content": user_message}],
        stream=True,
    )

    for chunk in stream:
        if hasattr(chunk, "error") and chunk.error:
            yield {"type": "error", "content": str(chunk.error)}
            return

        if not (chunk.event and chunk.event.payload):
            continue

        payload = chunk.event.payload

        if payload.event_type == "step_progress":
            delta = getattr(payload, "delta", None)
            if delta and getattr(delta, "type", None) == "text":
                yield {"type": "token", "content": delta.text}

        elif payload.event_type == "step_complete":
            step_type = getattr(payload, "step_type", None)

            if step_type == "tool_execution" and hasattr(payload.step_details, "tool_calls"):
                for tc in payload.step_details.tool_calls:
                    tool_name = str(getattr(tc, "tool_name", getattr(tc, "name", "")))
                    yield {"type": "tool_call", "tool": tool_name}

            elif step_type == "shield_call":
                violation = getattr(payload.step_details, "violation", None)
                if violation:
                    msg = getattr(violation, "user_message", "Blocked by safety policy")
                    logger.warning("  [Shield-Stream] BLOCKED: %s", msg)
                    yield {"type": "blocked", "content": msg}
                    yield {"type": "end"}
                    return
                else:
                    logger.info("  [Shield-Stream] PASSED (no violation)")

        elif payload.event_type == "turn_complete":
            pass  # final text already streamed via step_progress

    yield {"type": "end"}


# --- LangGraph: State and Nodes ---

class AgentState(TypedDict):
    query: str
    model: str
    input_blocked: bool
    block_message: str
    result: dict[str, Any] | None


def _extract_violated_categories(violation) -> set:
    """Extract Llama Guard category codes (S1-S13) from a SafetyViolation."""
    categories = set()
    # Check metadata for category info
    metadata = getattr(violation, "metadata", {}) or {}
    logger.info("  [Safety] Violation metadata: %s", metadata)
    for key, val in metadata.items():
        if isinstance(val, str):
            categories.update(re.findall(r"S\d+", val))
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, str):
                    categories.update(re.findall(r"S\d+", item))
    # Also check user_message for category codes
    user_msg = getattr(violation, "user_message", "") or ""
    categories.update(re.findall(r"S\d+", user_msg))
    return categories


def _is_violation_ignorable(violation) -> bool:
    """Check if all violated categories are in the ignore list."""
    categories = _extract_violated_categories(violation)
    if not categories:
        # No category info found — cannot determine, do not ignore
        return False
    ignored = categories.issubset(IGNORED_SAFETY_CATEGORIES)
    if ignored:
        logger.info("  [Safety] All violated categories %s are ignorable, allowing through", categories)
    return ignored


def input_safety_node(state: AgentState) -> dict:
    """Client-side content safety check (Llama Guard) on user input.
    Ignores violations from business-data categories (S5, S7, S8)."""
    query = state["query"]
    logger.info("  [InputSafety] Checking: %s", query[:80])

    try:
        result = client.safety.run_shield(
            shield_id="content_safety",
            messages=[{"role": "user", "content": query}],
            params={},
        )
        if result.violation:
            level = getattr(result.violation, "violation_level", "unknown")
            msg = getattr(result.violation, "user_message", "Request blocked by safety policy")
            logger.warning("  [InputSafety] Violation detected (level=%s): %s", level, msg)
            if not _is_violation_ignorable(result.violation):
                return {"input_blocked": True, "block_message": msg}
            logger.info("  [InputSafety] Violation ignored (business data false positive)")
    except Exception as e:
        logger.warning("  [InputSafety] Shield error (allowing through): %s", e)

    logger.info("  [InputSafety] PASSED")
    return {"input_blocked": False, "block_message": ""}


def agent_node(state: AgentState) -> dict:
    """Call the Llama Stack agent (handles tool calling + output shields internally)."""
    logger.info("  [Agent] Processing: %s (model: %s)", state["query"][:80], state.get("model", DEFAULT_MODEL))
    result = run_agent_turn(state["query"], model=state.get("model"))

    status = "BLOCKED" if result.get("blocked") else "ERROR" if result.get("error") else "OK"
    tools = len(result.get("tool_calls", []))
    logger.info("  [Agent] %s | tools used: %d", status, tools)

    return {"result": result}


def route_after_safety(state: AgentState) -> str:
    if state.get("input_blocked"):
        return "blocked"
    return "agent"


# Build the graph: input_safety -> (route) -> agent -> END
builder = StateGraph(AgentState)
builder.add_node("input_safety", input_safety_node)
builder.add_node("agent", agent_node)
builder.add_edge(START, "input_safety")
builder.add_conditional_edges("input_safety", route_after_safety, {
    "blocked": END,
    "agent": "agent",
})
builder.add_edge("agent", END)
graph = builder.compile()
logger.info("LangGraph compiled: START -> input_safety -> (route) -> agent -> END")


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


def extract_customer_and_data(tool_outputs: list, data_type: str = "orders"):
    """Extract customer info and orders/invoices from Llama Stack tool outputs."""
    customer_info = None
    data_list = []

    for output in tool_outputs:
        try:
            output_data = json.loads(output) if isinstance(output, str) else output

            if "results" in output_data and output_data.get("results"):
                customer_info = output_data["results"][0]

            if "data" in output_data and output_data.get("data"):
                data_list = output_data["data"]
            elif data_type in output_data and output_data.get(data_type):
                data_list = output_data[data_type]

        except (json.JSONDecodeError, TypeError):
            logger.warning("Could not parse tool output")

    return customer_info, data_list


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
            if ALLOWED_MODELS and m.identifier not in ALLOWED_MODELS:
                continue
            model_list.append({
                "identifier": m.identifier,
                "provider_id": getattr(m, "provider_id", None),
                "model_type": getattr(m, "model_type", None),
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
        "message": "Customer Orders and Invoices API (Llama Stack Toolgroup + Shields)",
        "endpoints": {
            "models": "/models  (list available models)",
            "find_orders": "/find_orders?email=<customer_email>&model=<model_id>  (SSE stream)",
            "find_invoices": "/find_invoices?email=<customer_email>&model=<model_id>  (SSE stream)",
            "question": "/question?q=<your_question>&model=<model_id>  (SSE stream)",
        },
        "features": {
            "toolgroups": ["mcp::customer", "mcp::finance", "builtin::websearch"],
            "input_shields": INPUT_SHIELDS,
            "output_shields": OUTPUT_SHIELDS,
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

            agent_result = result.get("result", {})
            if agent_result.get("blocked"):
                yield sse_event({"type": "blocked", "content": agent_result.get("shield_message", "Blocked")})
                yield sse_event({"type": "end"})
                return

            if agent_result.get("error"):
                yield sse_event({"type": "error", "content": agent_result["error"]})
                return

            customer_info, orders = extract_customer_and_data(
                agent_result.get("tool_outputs", []), "orders"
            )

            response = OrdersResponse(
                customer=Customer(**customer_info) if customer_info else None,
                orders=[Order(**order) for order in orders],
                total_orders=len(orders),
            )

            yield sse_event({"type": "result", "data": response.model_dump()})
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

            agent_result = result.get("result", {})
            if agent_result.get("blocked"):
                yield sse_event({"type": "blocked", "content": agent_result.get("shield_message", "Blocked")})
                yield sse_event({"type": "end"})
                return

            if agent_result.get("error"):
                yield sse_event({"type": "error", "content": agent_result["error"]})
                return

            customer_info, invoices = extract_customer_and_data(
                agent_result.get("tool_outputs", []), "invoices"
            )

            enriched_invoices = []
            for invoice in invoices:
                if customer_info:
                    invoice["customerId"] = invoice.get("customerId", customer_info.get("customerId"))
                    invoice["customerEmail"] = invoice.get("customerEmail", customer_info.get("contactEmail"))
                    invoice["contactName"] = customer_info.get("contactName")
                enriched_invoices.append(Invoice(**invoice))

            response = InvoicesResponse(
                customer=Customer(**customer_info) if customer_info else None,
                invoices=enriched_invoices,
                total_invoices=len(invoices),
            )

            yield sse_event({"type": "result", "data": response.model_dump()})
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
            try:
                safety_result = client.safety.run_shield(
                    shield_id="content_safety",
                    messages=[{"role": "user", "content": q}],
                    params={},
                )
                if safety_result.violation:
                    level = getattr(safety_result.violation, "violation_level", "unknown")
                    msg = getattr(safety_result.violation, "user_message", "Blocked by safety policy")
                    logger.warning("  [Question] Violation detected (level=%s): %s", level, msg)
                    if not _is_violation_ignorable(safety_result.violation):
                        yield sse_event({"type": "blocked", "content": msg})
                        yield sse_event({"type": "end"})
                        return
                    logger.info("  [Question] Violation ignored (business data false positive)")
                logger.info("  [Question] PASSED content_safety check")
            except Exception as e:
                logger.warning("  [Question] Safety check error (allowing through): %s", e)

            # Stream agent response
            has_content = False
            for event in run_agent_turn_streaming(q, model=model):
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

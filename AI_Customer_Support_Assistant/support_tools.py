"""
support_tools/support_tools.py
LangChain tools for order status and ticket creation.
"""

import json
import uuid
import logging
from datetime import datetime
from pathlib import Path

from langchain.tools import tool

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
ORDERS_FILE = DATA_DIR / "orders.json"
TICKETS_FILE = DATA_DIR / "tickets.json"


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


@tool
def check_order_status(order_id: str) -> str:
    """
    Check the status of a customer order by order ID.
    Use this when the customer asks about their order, delivery, tracking, or shipment.
    Input: order ID string (e.g. ORD123)
    """
    order_id = order_id.strip().upper()
    logger.info(f"[TOOL] check_order_status called with order_id={order_id}")

    try:
        orders = _load_json(ORDERS_FILE)
    except FileNotFoundError:
        logger.error("orders.json not found")
        return "Error: Order database is currently unavailable. Please try again later."

    if order_id not in orders:
        logger.warning(f"Order ID {order_id} not found")
        return f"Order ID '{order_id}' was not found. Please double-check and try again."

    order = orders[order_id]
    result = (
        f"Order Details for {order_id}:\n"
        f"  Customer         : {order.get('customer_name', 'N/A')}\n"
        f"  Status           : {order.get('status', 'unknown').capitalize()}\n"
        f"  Expected Delivery: {order.get('expected_delivery', 'N/A')}"
    )
    logger.info(f"[TOOL] check_order_status result: {result}")
    return result


@tool
def create_support_ticket(issue: str) -> str:
    """
    Create a support ticket for a customer issue.
    Use this when the customer reports a problem that needs follow-up —
    such as a missing package, damaged item, or refund request.
    Input: description of the customer's issue.
    """
    logger.info(f"[TOOL] create_support_ticket called: '{issue[:80]}'")

    try:
        tickets = _load_json(TICKETS_FILE)
    except (FileNotFoundError, json.JSONDecodeError):
        tickets = []

    ticket_id = f"TKT-{uuid.uuid4().hex[:8].upper()}"
    ticket = {
        "ticket_id": ticket_id,
        "issue": issue,
        "status": "open",
        "created_at": datetime.now().isoformat(),
    }
    tickets.append(ticket)

    try:
        _save_json(TICKETS_FILE, tickets)
    except Exception as e:
        logger.error(f"Failed to save ticket: {e}")
        return "Error: Could not create support ticket. Please try again later."

    result = (
        f"Support ticket created successfully!\n"
        f"  Ticket ID : {ticket_id}\n"
        f"  Status    : Open\n"
        f"  Issue     : {issue}\n"
        f"Our team will reach out within 24 hours."
    )
    logger.info(f"[TOOL] Ticket created: {ticket_id}")
    return result


ALL_TOOLS = [check_order_status, create_support_ticket]

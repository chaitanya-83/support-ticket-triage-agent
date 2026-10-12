from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class Category(str, Enum):
    BILLING = "billing"
    REFUND = "refund"
    DAMAGED = "damaged"
    DELIVERY = "delivery"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    OTHER = "other"


class Classification(BaseModel):
    """LLM output for the classify node."""
    category: Category
    reason: str = Field(description="One short sentence explaining the category")


class PriorityResult(BaseModel):
    """LLM output for the priority node."""
    priority: int = Field(ge=1, le=4, description="1=critical, 2=high, 3=medium, 4=low")
    reason: str = Field(description="One short sentence explaining the priority")


# ---------- API models ----------
class TicketIn(BaseModel):
    text: str = Field(min_length=5, examples=["I was charged twice and need a refund urgently"])
    customer: Optional[str] = "anonymous"


class TicketOut(BaseModel):
    ticket_id: int
    category: Category
    priority: int
    assigned_agent: str
    assignment_reason: str
    trace: List[str]

"""Mutable state owned by one user request, never by the shared agent."""

from dataclasses import dataclass, field
from typing import Literal

from pydantic_ai.messages import ModelRequest

from .accounting import RequestAccounting
from .evidence import EvidenceLedger


@dataclass
class ResearchRunState:
    run_id: str | None = None
    phase: Literal["research", "repair"] = "research"
    research_requests: int = 0
    summary_requests: int = 0
    citation_repairs: int = 0
    summary_disabled: bool = False
    compactions: int = 0
    recovery_requests: int = 0
    question: ModelRequest | None = None
    summary: str = ""
    ledger: EvidenceLedger = field(default_factory=EvidenceLedger)
    accounting: RequestAccounting = field(default_factory=RequestAccounting)

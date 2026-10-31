"""Persistent intermediate state for the unified pipeline."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases import (
    Phase1State,
    Phase2State,
    Phase3State,
    Phase4State,
    Phase5State,
    Phase6State,
    Phase7State,
    Phase8State,
)


class MLPrepPipelineState(BaseModel):
    """States accumulated while executing Phases 1 through 8."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    state_version: str = "1"
    last_completed_phase: str | None = None
    phase1_state: Phase1State | None = None
    phase2_state: Phase2State | None = None
    phase3_state: Phase3State | None = None
    phase4_state: Phase4State | None = None
    phase5_state: Phase5State | None = None
    phase6_state: Phase6State | None = None
    phase7_state: Phase7State | None = None
    phase8_state: Phase8State | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

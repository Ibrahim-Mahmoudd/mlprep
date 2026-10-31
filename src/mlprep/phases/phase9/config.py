"""Configuration for the Phase 9 model handoff."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Phase9Config:
    """Controls validation and copying at the handoff boundary."""

    require_disjoint_rows: bool = True
    copy_partitions: bool = True

    def __post_init__(self) -> None:
        """Validate configuration flags."""

        for name in (
            "require_disjoint_rows",
            "copy_partitions",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean.")

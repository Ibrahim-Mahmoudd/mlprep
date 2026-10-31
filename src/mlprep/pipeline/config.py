"""Configuration for the unified nine-phase pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from mlprep.phases import (
    Phase1Config,
    Phase2Config,
    Phase3Config,
    Phase4Config,
    Phase5Config,
    Phase6Config,
    Phase7Config,
    Phase8Config,
    Phase9Config,
)


@dataclass(slots=True)
class MLPrepPipelineConfig:
    """Resolved configuration for one complete pipeline execution."""

    phase1: Phase1Config
    phase2: Phase2Config
    phase3: Phase3Config = field(default_factory=Phase3Config)
    phase4: Phase4Config = field(default_factory=Phase4Config)
    phase5: Phase5Config = field(default_factory=Phase5Config)
    phase6: Phase6Config = field(default_factory=Phase6Config)
    phase7: Phase7Config = field(default_factory=Phase7Config)
    phase8: Phase8Config = field(default_factory=Phase8Config)
    phase9: Phase9Config = field(default_factory=Phase9Config)
    output_directory: Path | str | None = None
    save_intermediate_states: bool = True

    def __post_init__(self) -> None:
        """Validate phase configuration types and output settings."""

        expected_types = (
            ("phase1", self.phase1, Phase1Config),
            ("phase2", self.phase2, Phase2Config),
            ("phase3", self.phase3, Phase3Config),
            ("phase4", self.phase4, Phase4Config),
            ("phase5", self.phase5, Phase5Config),
            ("phase6", self.phase6, Phase6Config),
            ("phase7", self.phase7, Phase7Config),
            ("phase8", self.phase8, Phase8Config),
            ("phase9", self.phase9, Phase9Config),
        )
        for name, value, expected_type in expected_types:
            if not isinstance(value, expected_type):
                raise TypeError(
                    f"{name} must be an instance of "
                    f"{expected_type.__name__}."
                )

        if self.output_directory is not None:
            if not isinstance(self.output_directory, (Path, str)):
                raise TypeError(
                    "output_directory must be a pathlib.Path, string, or None."
                )
            if not str(self.output_directory).strip():
                raise ValueError("output_directory cannot be empty.")
            self.output_directory = Path(self.output_directory)

        if not isinstance(self.save_intermediate_states, bool):
            raise TypeError("save_intermediate_states must be a boolean.")

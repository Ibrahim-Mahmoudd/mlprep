"""Configuration for the unified nine-phase pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from mlprep.cross_validation.config import CrossValidationConfig
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


@dataclass(slots=True)
class PipelineOptions:
    """Output and checkpoint controls within the unified config."""

    output_directory: Path | str | None = None
    save_intermediate_states: bool = True

    def __post_init__(self) -> None:
        if self.output_directory is not None:
            if not isinstance(self.output_directory, (Path, str)):
                raise TypeError(
                    "pipeline.output_directory must be a path, string, or None."
                )
            if not str(self.output_directory).strip():
                raise ValueError("pipeline.output_directory cannot be empty.")
            self.output_directory = Path(self.output_directory)
        if not isinstance(self.save_intermediate_states, bool):
            raise TypeError(
                "pipeline.save_intermediate_states must be a boolean."
            )


@dataclass(slots=True)
class MLPrepConfig(MLPrepPipelineConfig):
    """Unified, versioned configuration for phases, CV, and the runner."""

    schema_version: str = "1"
    cross_validation: CrossValidationConfig = field(
        default_factory=CrossValidationConfig
    )
    pipeline: PipelineOptions = field(default_factory=PipelineOptions)

    def __post_init__(self) -> None:
        MLPrepPipelineConfig.__post_init__(self)
        if self.schema_version != "1":
            raise ValueError(
                f"Unsupported MLPrep config version: "
                f"{self.schema_version!r}."
            )
        if not isinstance(self.cross_validation, CrossValidationConfig):
            raise TypeError(
                "cross_validation must be a CrossValidationConfig."
            )
        if not isinstance(self.pipeline, PipelineOptions):
            raise TypeError("pipeline must be a PipelineOptions instance.")

        legacy_output = self.output_directory
        pipeline_output = self.pipeline.output_directory
        if (
            legacy_output is not None
            and pipeline_output is not None
            and legacy_output != pipeline_output
        ):
            raise ValueError(
                "output_directory conflicts with pipeline.output_directory."
            )
        resolved_output = pipeline_output or legacy_output
        self.output_directory = resolved_output
        self.pipeline.output_directory = resolved_output

        if not self.save_intermediate_states:
            self.pipeline.save_intermediate_states = False
        self.save_intermediate_states = (
            self.pipeline.save_intermediate_states
        )

    @property
    def config_version(self) -> str:
        """Return the canonical configuration version."""

        return self.schema_version

    @property
    def resolved_configuration(self) -> dict[str, object]:
        """Return the full configuration in the public flat schema."""

        from mlprep.pipeline.execution_manifest import (
            PipelineExecutionManifest,
        )

        config: dict[str, object] = {
            "config_version": self.config_version,
        }
        for phase_number in range(1, 10):
            phase_name = f"phase{phase_number}"
            config[phase_name] = (
                PipelineExecutionManifest.configuration_value(
                    getattr(self, phase_name)
                )
            )
        config["cross_validation"] = (
            PipelineExecutionManifest.configuration_value(
                self.cross_validation
            )
        )
        config["pipeline"] = (
            PipelineExecutionManifest.configuration_value(self.pipeline)
        )
        return config

    @property
    def configuration_fingerprint(self) -> str:
        """Return a stable fingerprint of the resolved config."""

        from mlprep.pipeline.execution_manifest import (
            PipelineExecutionManifest,
        )

        return PipelineExecutionManifest.fingerprint_configuration(
            self.resolved_configuration
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize the resolved configuration in its public schema."""

        return self.resolved_configuration

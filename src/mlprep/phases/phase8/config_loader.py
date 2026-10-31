"""Configuration loading utilities for Phase 8."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import yaml

from mlprep.phases.phase8.analysis import (
    ClassDistributionOptions,
    ImbalanceMetricsOptions,
    StrategyRecommenderOptions,
)
from mlprep.phases.phase8.config import (
    Phase8Config,
    Phase8ExecutionOptions,
    Phase8ValidationOptions,
)
from mlprep.phases.phase8.contract import (
    BalancingContractOptions,
    BalancingPlanOptions,
    BalancingStrategy,
    BalancingTask,
    ClassWeightMethod,
    HybridCleaningKind,
    NeighborShortagePolicy,
    SamplingStrategyKind,
    WeightNormalizationKind,
)
from mlprep.phases.phase8.orchestration import (
    BalancingPipelineOptions,
    PartitionGuardOptions,
)
from mlprep.phases.phase8.output_governance import (
    Phase8ReadinessOptions,
    Phase8ReportOptions,
)
from mlprep.phases.phase8.validation import (
    DistributionValidationOptions,
    PartitionIntegrityOptions,
    SchemaConsistencyOptions,
    SyntheticIntegrityOptions,
    TargetIntegrityOptions,
)
from mlprep.phases.phase8.weighting import (
    ClassWeightOptions,
    SampleWeightOptions,
)


ConfigFormat = Literal[
    "yaml",
    "json",
]


class Phase8ConfigLoader:
    """Load, merge, validate, and save Phase 8 configurations."""

    supported_extensions = {
        ".yaml",
        ".yml",
        ".json",
    }

    allowed_sections = {
        "execution",
        "contract",
        "plan",
        "distribution_analysis",
        "imbalance_metrics",
        "strategy_recommender",
        "class_weights",
        "sample_weights",
        "partition_guard",
        "pipeline",
        "validation",
        "readiness",
        "reporting",
    }

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        overrides: Mapping[str, Any] | None = None,
        dotted_overrides: Mapping[str, Any] | None = None,
    ) -> Phase8Config:
        """Load a Phase 8 configuration."""

        values = cls.load_raw(
            path
        )

        if overrides is not None:
            values = cls.merge(
                base=values,
                override=overrides,
            )

        if dotted_overrides is not None:
            values = cls.apply_dot_overrides(
                values=values,
                overrides=dotted_overrides,
            )

        return cls.from_dict(
            values
        )

    @classmethod
    def load_raw(
        cls,
        path: str | Path,
    ) -> dict[str, Any]:
        """Load a raw YAML or JSON configuration."""

        input_path = Path(
            path
        )

        if not input_path.exists():
            raise FileNotFoundError(
                f"Configuration does not exist: {input_path}"
            )

        if not input_path.is_file():
            raise IsADirectoryError(
                f"Configuration path is not a file: {input_path}"
            )

        extension = input_path.suffix.casefold()

        if extension not in cls.supported_extensions:
            raise ValueError(
                "Unsupported configuration extension: "
                f"{extension!r}"
            )

        content = input_path.read_text(
            encoding="utf-8"
        )

        if extension in {
            ".yaml",
            ".yml",
        }:
            try:
                values = yaml.safe_load(
                    content
                )

            except yaml.YAMLError as error:
                raise ValueError(
                    f"Invalid YAML configuration: {error}"
                ) from error

        else:
            try:
                values = json.loads(
                    content
                )

            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON configuration: {error}"
                ) from error

        if values is None:
            raise ValueError(
                f"Configuration is empty: {input_path}"
            )

        if not isinstance(
            values,
            dict,
        ):
            raise TypeError(
                "Configuration root must be a mapping."
            )

        return values

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Any],
    ) -> Phase8Config:
        """Create Phase 8 configuration from a mapping."""

        root = deepcopy(
            dict(
                values
            )
        )

        unknown_sections = set(
            root
        ).difference(
            cls.allowed_sections
        )

        if unknown_sections:
            raise ValueError(
                "Unknown Phase 8 configuration sections: "
                f"{sorted(unknown_sections)}"
            )

        distribution = ClassDistributionOptions(
            **cls._mapping(
                root.get(
                    "distribution_analysis",
                    {},
                ),
                section="distribution_analysis",
            )
        )

        metrics = ImbalanceMetricsOptions(
            **cls._mapping(
                root.get(
                    "imbalance_metrics",
                    {},
                ),
                section="imbalance_metrics",
            )
        )

        recommender = cls._build_recommender(
            root.get(
                "strategy_recommender",
                {},
            )
        )

        class_weights = cls._build_class_weights(
            root.get(
                "class_weights",
                {},
            )
        )

        sample_weights = SampleWeightOptions(
            **cls._mapping(
                root.get(
                    "sample_weights",
                    {},
                ),
                section="sample_weights",
            )
        )

        partition_guard = PartitionGuardOptions(
            **cls._mapping(
                root.get(
                    "partition_guard",
                    {},
                ),
                section="partition_guard",
            )
        )

        pipeline_data = cls._mapping(
            root.get(
                "pipeline",
                {},
            ),
            section="pipeline",
        )

        for child_name in (
            "distribution",
            "metrics",
            "recommender",
            "class_weights",
            "sample_weights",
            "partition_guard",
        ):
            pipeline_data.pop(
                child_name,
                None,
            )

        pipeline = BalancingPipelineOptions(
            distribution=distribution,
            metrics=metrics,
            recommender=recommender,
            class_weights=class_weights,
            sample_weights=sample_weights,
            partition_guard=partition_guard,
            **pipeline_data,
        )

        return Phase8Config(
            execution=Phase8ExecutionOptions(
                **cls._mapping(
                    root.get(
                        "execution",
                        {},
                    ),
                    section="execution",
                )
            ),
            contract=BalancingContractOptions(
                **cls._mapping(
                    root.get(
                        "contract",
                        {},
                    ),
                    section="contract",
                )
            ),
            plan=cls._build_plan(
                root.get(
                    "plan",
                    {},
                )
            ),
            distribution_analysis=distribution,
            imbalance_metrics=metrics,
            strategy_recommender=recommender,
            class_weights=class_weights,
            sample_weights=sample_weights,
            partition_guard=partition_guard,
            pipeline=pipeline,
            validation=cls._build_validation(
                root.get(
                    "validation",
                    {},
                )
            ),
            readiness=Phase8ReadinessOptions(
                **cls._mapping(
                    root.get(
                        "readiness",
                        {},
                    ),
                    section="readiness",
                )
            ),
            reporting=Phase8ReportOptions(
                **cls._mapping(
                    root.get(
                        "reporting",
                        {},
                    ),
                    section="reporting",
                )
            ),
        )

    @classmethod
    def _build_plan(
        cls,
        values: object,
    ) -> BalancingPlanOptions:
        """Build balancing-plan options."""

        data = cls._mapping(
            values,
            section="plan",
        )

        enum_fields = {
            "task": BalancingTask,
            "strategy": BalancingStrategy,
            "sampling_strategy": SamplingStrategyKind,
            "neighbor_shortage_policy": (
                NeighborShortagePolicy
            ),
            "hybrid_cleaning": HybridCleaningKind,
            "class_weight_method": ClassWeightMethod,
            "class_weight_normalization": (
                WeightNormalizationKind
            ),
        }

        for field_name, enum_type in enum_fields.items():
            if (
                field_name in data
                and data[
                    field_name
                ] is not None
            ):
                data[
                    field_name
                ] = enum_type(
                    data[
                        field_name
                    ]
                )

        return BalancingPlanOptions(
            **data
        )

    @classmethod
    def _build_recommender(
        cls,
        values: object,
    ) -> StrategyRecommenderOptions:
        """Build strategy-recommender options."""

        data = cls._mapping(
            values,
            section="strategy_recommender",
        )

        strategy_fields = {
            "mild_imbalance_strategy",
            "moderate_imbalance_strategy",
            "severe_imbalance_strategy",
            "extreme_imbalance_strategy",
            "neighbor_shortage_fallback",
        }

        for field_name in strategy_fields:
            if field_name in data:
                data[
                    field_name
                ] = BalancingStrategy(
                    data[
                        field_name
                    ]
                )

        return StrategyRecommenderOptions(
            **data
        )

    @classmethod
    def _build_class_weights(
        cls,
        values: object,
    ) -> ClassWeightOptions:
        """Build class-weight options."""

        data = cls._mapping(
            values,
            section="class_weights",
        )

        if "method" in data:
            data[
                "method"
            ] = ClassWeightMethod(
                data[
                    "method"
                ]
            )

        if "normalization" in data:
            data[
                "normalization"
            ] = WeightNormalizationKind(
                data[
                    "normalization"
                ]
            )

        return ClassWeightOptions(
            **data
        )

    @classmethod
    def _build_validation(
        cls,
        values: object,
    ) -> Phase8ValidationOptions:
        """Build Phase 8 validation options."""

        data = cls._mapping(
            values,
            section="validation",
        )

        allowed = {
            "distribution",
            "schema_consistency",
            "synthetic_integrity",
            "target_integrity",
            "partition_integrity",
        }

        unknown = set(
            data
        ).difference(
            allowed
        )

        if unknown:
            raise ValueError(
                "Unknown Phase 8 validation options: "
                f"{sorted(unknown)}"
            )

        return Phase8ValidationOptions(
            distribution=DistributionValidationOptions(
                **cls._mapping(
                    data.get(
                        "distribution",
                        {},
                    ),
                    section="validation.distribution",
                )
            ),
            schema_consistency=SchemaConsistencyOptions(
                **cls._mapping(
                    data.get(
                        "schema_consistency",
                        {},
                    ),
                    section=(
                        "validation.schema_consistency"
                    ),
                )
            ),
            synthetic_integrity=SyntheticIntegrityOptions(
                **cls._mapping(
                    data.get(
                        "synthetic_integrity",
                        {},
                    ),
                    section=(
                        "validation.synthetic_integrity"
                    ),
                )
            ),
            target_integrity=TargetIntegrityOptions(
                **cls._mapping(
                    data.get(
                        "target_integrity",
                        {},
                    ),
                    section=(
                        "validation.target_integrity"
                    ),
                )
            ),
            partition_integrity=PartitionIntegrityOptions(
                **cls._mapping(
                    data.get(
                        "partition_integrity",
                        {},
                    ),
                    section=(
                        "validation.partition_integrity"
                    ),
                )
            ),
        )

    @classmethod
    def save(
        cls,
        config: Phase8Config,
        path: str | Path,
        *,
        config_format: ConfigFormat | None = None,
        overwrite: bool = False,
    ) -> Path:
        """Save a resolved Phase 8 configuration."""

        if not isinstance(
            config,
            Phase8Config,
        ):
            raise TypeError(
                "config must be a Phase8Config."
            )

        output_path = Path(
            path
        )

        if (
            output_path.exists()
            and not overwrite
        ):
            raise FileExistsError(
                f"Configuration already exists: {output_path}"
            )

        resolved_format = (
            config_format
            or cls._format_from_extension(
                output_path
            )
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        values = config.to_dict()

        if resolved_format == "yaml":
            content = yaml.safe_dump(
                values,
                sort_keys=False,
                allow_unicode=True,
            )

        else:
            content = json.dumps(
                values,
                indent=2,
                ensure_ascii=False,
                default=str,
            )

        output_path.write_text(
            content,
            encoding="utf-8",
        )

        return output_path

    @classmethod
    def merge(
        cls,
        *,
        base: Mapping[str, Any],
        override: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Recursively merge configuration mappings."""

        merged = deepcopy(
            dict(
                base
            )
        )

        for key, override_value in override.items():
            existing_value = merged.get(
                key
            )

            if (
                isinstance(
                    existing_value,
                    Mapping,
                )
                and isinstance(
                    override_value,
                    Mapping,
                )
            ):
                merged[
                    key
                ] = cls.merge(
                    base=existing_value,
                    override=override_value,
                )

            else:
                merged[
                    key
                ] = deepcopy(
                    override_value
                )

        return merged

    @classmethod
    def apply_dot_overrides(
        cls,
        *,
        values: Mapping[str, Any],
        overrides: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Apply dotted configuration overrides."""

        updated = deepcopy(
            dict(
                values
            )
        )

        for dotted_path, value in overrides.items():
            cls._set_dotted_value(
                values=updated,
                dotted_path=dotted_path,
                value=value,
            )

        return updated

    @staticmethod
    def _mapping(
        value: object,
        *,
        section: str,
    ) -> dict[str, Any]:
        """Validate and copy one mapping."""

        if value is None:
            return {}

        if not isinstance(
            value,
            Mapping,
        ):
            raise TypeError(
                f"Configuration section {section!r} "
                "must be a mapping."
            )

        return deepcopy(
            dict(
                value
            )
        )

    @staticmethod
    def _format_from_extension(
        path: Path,
    ) -> ConfigFormat:
        """Resolve an output format from the extension."""

        extension = path.suffix.casefold()

        if extension in {
            ".yaml",
            ".yml",
        }:
            return "yaml"

        if extension == ".json":
            return "json"

        raise ValueError(
            "Cannot determine configuration format "
            f"from extension {extension!r}."
        )

    @staticmethod
    def _set_dotted_value(
        *,
        values: dict[str, Any],
        dotted_path: str,
        value: Any,
    ) -> None:
        """Set a nested value using a dotted path."""

        parts = dotted_path.split(
            "."
        )

        if (
            not dotted_path.strip()
            or any(
                not part.strip()
                for part in parts
            )
        ):
            raise ValueError(
                f"Invalid dotted path: {dotted_path!r}"
            )

        current = values

        for part in parts[
            :-1
        ]:
            existing = current.get(
                part
            )

            if existing is None:
                created: dict[
                    str,
                    Any,
                ] = {}

                current[
                    part
                ] = created

                current = created

                continue

            if not isinstance(
                existing,
                dict,
            ):
                raise TypeError(
                    f"Cannot set {dotted_path!r}; "
                    f"{part!r} is not a mapping."
                )

            current = existing

        current[
            parts[
                -1
            ]
        ] = deepcopy(
            value
        )
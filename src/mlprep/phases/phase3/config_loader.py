"""Configuration loading utilities for Phase 3."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, ClassVar, Literal

import yaml

from mlprep.phases.phase3.config import (
    Phase3Config,
    Phase3DiagnosticOptions,
    Phase3ExecutionOptions,
    Phase3ProfilingOptions,
    Phase3RelationshipOptions,
)
from mlprep.phases.phase3.contract.column_selection import (
    ColumnSelectionOptions,
)
from mlprep.phases.phase3.contract.eda_contract import (
    EDAContractOptions,
)
from mlprep.phases.phase3.diagnostics.data_quality import (
    DataQualityDiagnosticOptions,
)
from mlprep.phases.phase3.diagnostics.distribution_shift import (
    DistributionShiftOptions,
)
from mlprep.phases.phase3.diagnostics.high_cardinality import (
    HighCardinalityDiagnosticOptions,
)
from mlprep.phases.phase3.diagnostics.rare_categories import (
    RareCategoryDiagnosticOptions,
)
from mlprep.phases.phase3.diagnostics.skewness import (
    SkewnessDiagnosticOptions,
)
from mlprep.phases.phase3.diagnostics.temporal_patterns import (
    TemporalPatternOptions,
)
from mlprep.phases.phase3.output_governance.readiness import (
    Phase3ReadinessOptions,
)
from mlprep.phases.phase3.output_governance.report import (
    Phase3ReportOptions,
)
from mlprep.phases.phase3.profiling.categorical_profile import (
    CategoricalProfilerOptions,
)
from mlprep.phases.phase3.profiling.dataset_profile import (
    DatasetProfilerOptions,
)
from mlprep.phases.phase3.profiling.datetime_profile import (
    DatetimeProfilerOptions,
)
from mlprep.phases.phase3.profiling.missingness_profile import (
    MissingnessProfilerOptions,
)
from mlprep.phases.phase3.profiling.numeric_profile import (
    NumericProfilerOptions,
)
from mlprep.phases.phase3.profiling.target_profile import (
    TargetProfilerOptions,
)
from mlprep.phases.phase3.profiling.text_profile import (
    TextProfilerOptions,
)
from mlprep.phases.phase3.recommendations.encoding_recommendations import (
    EncodingRecommendationOptions,
)
from mlprep.phases.phase3.recommendations.imputation_recommendations import (
    ImputationRecommendationOptions,
)
from mlprep.phases.phase3.recommendations.interaction_recommendations import (
    InteractionRecommendationOptions,
)
from mlprep.phases.phase3.recommendations.outlier_recommendations import (
    OutlierRecommendationOptions,
)
from mlprep.phases.phase3.recommendations.recommendation_engine import (
    RecommendationEngineOptions,
)
from mlprep.phases.phase3.recommendations.transformation_recommendations import (
    TransformationRecommendationOptions,
)
from mlprep.phases.phase3.relationships.categorical_associations import (
    CategoricalAssociationOptions,
)
from mlprep.phases.phase3.relationships.interaction_candidates import (
    InteractionCandidateOptions,
)
from mlprep.phases.phase3.relationships.missingness_associations import (
    MissingnessAssociationOptions,
)
from mlprep.phases.phase3.relationships.mixed_associations import (
    MixedAssociationOptions,
)
from mlprep.phases.phase3.relationships.multicollinearity import (
    MulticollinearityOptions,
)
from mlprep.phases.phase3.relationships.numeric_associations import (
    NumericAssociationOptions,
)
from mlprep.phases.phase3.relationships.target_associations import (
    TargetAssociationOptions,
)
from mlprep.phases.phase3.visualization.categorical_plots import (
    CategoricalPlotOptions,
)
from mlprep.phases.phase3.visualization.missingness_plots import (
    MissingnessPlotOptions,
)
from mlprep.phases.phase3.visualization.numeric_plots import (
    NumericPlotOptions,
)
from mlprep.phases.phase3.visualization.plot_manager import (
    PlotManagerOptions,
)
from mlprep.phases.phase3.visualization.relationship_plots import (
    RelationshipPlotOptions,
)
from mlprep.phases.phase3.visualization.target_plots import (
    TargetPlotOptions,
)


ConfigFormat = Literal[
    "yaml",
    "json",
]


class Phase3ConfigLoader:
    """Load, merge, validate, and save Phase 3 configurations."""

    yaml_extensions: ClassVar[frozenset[str]] = frozenset({
        ".yaml",
        ".yml",
    })

    json_extensions: ClassVar[frozenset[str]] = frozenset({
        ".json",
    })

    supported_extensions: ClassVar[frozenset[str]] = (
        yaml_extensions
        | json_extensions
    )

    allowed_sections: ClassVar[frozenset[str]] = frozenset({
        "execution",
        "contract",
        "column_selection",
        "profiling",
        "relationships",
        "diagnostics",
        "visualizations",
        "recommendations",
        "readiness",
        "reporting",
    })

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        overrides: Mapping[str, Any] | None = None,
        dotted_overrides: Mapping[
            str,
            Any,
        ] | None = None,
    ) -> Phase3Config:
        """Load Phase 3 configuration from YAML or JSON."""

        values = cls.load_raw(path)

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

        return cls.from_dict(values)

    @classmethod
    def load_raw(
        cls,
        path: str | Path,
    ) -> dict[str, Any]:
        """Load raw configuration values."""

        input_path = cls._validate_input_path(
            path
        )

        extension = input_path.suffix.casefold()

        if extension in cls.yaml_extensions:
            return cls._load_yaml(
                input_path
            )

        if extension in cls.json_extensions:
            return cls._load_json(
                input_path
            )

        supported_extensions = ", ".join(
            sorted(cls.supported_extensions)
        )

        raise ValueError(
            f"Unsupported configuration extension "
            f"{extension!r}. Supported extensions: "
            f"{supported_extensions}"
        )

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Any],
    ) -> Phase3Config:
        """Create Phase 3 configuration from a mapping."""

        root = deepcopy(
            dict(values)
        )

        unknown_sections = set(
            root
        ).difference(
            cls.allowed_sections
        )

        if unknown_sections:
            unknown_text = ", ".join(
                sorted(unknown_sections)
            )

            raise ValueError(
                "Unknown Phase 3 configuration sections: "
                f"{unknown_text}"
            )

        execution = Phase3ExecutionOptions(
            **cls._mapping(
                root.get(
                    "execution",
                    {},
                ),
                section="execution",
            )
        )

        contract = EDAContractOptions(
            **cls._mapping(
                root.get(
                    "contract",
                    {},
                ),
                section="contract",
            )
        )

        column_selection = ColumnSelectionOptions(
            **cls._mapping(
                root.get(
                    "column_selection",
                    {},
                ),
                section="column_selection",
            )
        )

        profiling = cls._build_profiling(
            root.get(
                "profiling",
                {},
            )
        )

        relationships = cls._build_relationships(
            root.get(
                "relationships",
                {},
            )
        )

        diagnostics = cls._build_diagnostics(
            root.get(
                "diagnostics",
                {},
            )
        )

        visualizations = cls._build_visualizations(
            root.get(
                "visualizations",
                {},
            )
        )

        recommendations = (
            cls._build_recommendations(
                root.get(
                    "recommendations",
                    {},
                )
            )
        )

        readiness = Phase3ReadinessOptions(
            **cls._mapping(
                root.get(
                    "readiness",
                    {},
                ),
                section="readiness",
            )
        )

        reporting = Phase3ReportOptions(
            **cls._mapping(
                root.get(
                    "reporting",
                    {},
                ),
                section="reporting",
            )
        )

        return Phase3Config(
            execution=execution,
            contract=contract,
            column_selection=column_selection,
            profiling=profiling,
            relationships=relationships,
            diagnostics=diagnostics,
            visualizations=visualizations,
            recommendations=recommendations,
            readiness=readiness,
            reporting=reporting,
        )

    @classmethod
    def save(
        cls,
        config: Phase3Config,
        path: str | Path,
        *,
        config_format: ConfigFormat | None = None,
        overwrite: bool = False,
    ) -> Path:
        """Save resolved Phase 3 configuration."""

        if not isinstance(config, Phase3Config):
            raise TypeError(
                "config must be an instance of Phase3Config."
            )

        output_path = Path(path)

        if (
            output_path.exists()
            and not overwrite
        ):
            raise FileExistsError(
                f"Configuration file already exists: "
                f"{output_path}"
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
    def validate_file(
        cls,
        path: str | Path,
    ) -> list[object]:
        """Validate a configuration file and return errors."""

        errors: list[str] = []

        try:
            cls.load(path)

        except (
            FileNotFoundError,
            IsADirectoryError,
            PermissionError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
            yaml.YAMLError,
        ) as error:
            errors.append(
                str(error)
            )

        return errors

    @classmethod
    def merge(
        cls,
        *,
        base: Mapping[str, Any],
        override: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Recursively merge configuration mappings."""

        merged = deepcopy(
            dict(base)
        )

        for key, override_value in (
            override.items()
        ):
            existing_value = merged.get(key)

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
                merged[key] = cls.merge(
                    base=existing_value,
                    override=override_value,
                )

            else:
                merged[key] = deepcopy(
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
        """Apply overrides using dotted configuration paths."""

        updated_values = deepcopy(
            dict(values)
        )

        for dotted_path, value in (
            overrides.items()
        ):
            cls._set_dotted_value(
                values=updated_values,
                dotted_path=dotted_path,
                value=value,
            )

        return updated_values

    @classmethod
    def _build_profiling(
        cls,
        values: object,
    ) -> Phase3ProfilingOptions:
        """Build profiling options."""

        profiling_values = cls._mapping(
            values,
            section="profiling",
        )

        cls._reject_unknown_keys(
            values=profiling_values,
            allowed={
                "dataset",
                "numeric",
                "categorical",
                "datetime",
                "text",
                "missingness",
                "target",
            },
            section="profiling",
        )

        return Phase3ProfilingOptions(
            dataset=DatasetProfilerOptions(
                **cls._mapping(
                    profiling_values.get(
                        "dataset",
                        {},
                    ),
                    section="profiling.dataset",
                )
            ),
            numeric=NumericProfilerOptions(
                **cls._mapping(
                    profiling_values.get(
                        "numeric",
                        {},
                    ),
                    section="profiling.numeric",
                )
            ),
            categorical=(
                CategoricalProfilerOptions(
                    **cls._mapping(
                        profiling_values.get(
                            "categorical",
                            {},
                        ),
                        section=(
                            "profiling.categorical"
                        ),
                    )
                )
            ),
            datetime=DatetimeProfilerOptions(
                **cls._mapping(
                    profiling_values.get(
                        "datetime",
                        {},
                    ),
                    section="profiling.datetime",
                )
            ),
            text=TextProfilerOptions(
                **cls._mapping(
                    profiling_values.get(
                        "text",
                        {},
                    ),
                    section="profiling.text",
                )
            ),
            missingness=MissingnessProfilerOptions(
                **cls._mapping(
                    profiling_values.get(
                        "missingness",
                        {},
                    ),
                    section=(
                        "profiling.missingness"
                    ),
                )
            ),
            target=TargetProfilerOptions(
                **cls._mapping(
                    profiling_values.get(
                        "target",
                        {},
                    ),
                    section="profiling.target",
                )
            ),
        )

    @classmethod
    def _build_relationships(
        cls,
        values: object,
    ) -> Phase3RelationshipOptions:
        """Build relationship-analysis options."""

        relationship_values = cls._mapping(
            values,
            section="relationships",
        )

        cls._reject_unknown_keys(
            values=relationship_values,
            allowed={
                "numeric",
                "categorical",
                "mixed",
                "target",
                "missingness",
                "multicollinearity",
                "interactions",
            },
            section="relationships",
        )

        return Phase3RelationshipOptions(
            numeric=NumericAssociationOptions(
                **cls._mapping(
                    relationship_values.get(
                        "numeric",
                        {},
                    ),
                    section=(
                        "relationships.numeric"
                    ),
                )
            ),
            categorical=(
                CategoricalAssociationOptions(
                    **cls._mapping(
                        relationship_values.get(
                            "categorical",
                            {},
                        ),
                        section=(
                            "relationships.categorical"
                        ),
                    )
                )
            ),
            mixed=MixedAssociationOptions(
                **cls._mapping(
                    relationship_values.get(
                        "mixed",
                        {},
                    ),
                    section=(
                        "relationships.mixed"
                    ),
                )
            ),
            target=TargetAssociationOptions(
                **cls._mapping(
                    relationship_values.get(
                        "target",
                        {},
                    ),
                    section=(
                        "relationships.target"
                    ),
                )
            ),
            missingness=(
                MissingnessAssociationOptions(
                    **cls._mapping(
                        relationship_values.get(
                            "missingness",
                            {},
                        ),
                        section=(
                            "relationships.missingness"
                        ),
                    )
                )
            ),
            multicollinearity=MulticollinearityOptions(
                **cls._mapping(
                    relationship_values.get(
                        "multicollinearity",
                        {},
                    ),
                    section=(
                        "relationships.multicollinearity"
                    ),
                )
            ),
            interactions=InteractionCandidateOptions(
                **cls._mapping(
                    relationship_values.get(
                        "interactions",
                        {},
                    ),
                    section=(
                        "relationships.interactions"
                    ),
                )
            ),
        )

    @classmethod
    def _build_diagnostics(
        cls,
        values: object,
    ) -> Phase3DiagnosticOptions:
        """Build diagnostic options."""

        diagnostic_values = cls._mapping(
            values,
            section="diagnostics",
        )

        cls._reject_unknown_keys(
            values=diagnostic_values,
            allowed={
                "skewness",
                "rare_categories",
                "high_cardinality",
                "distribution_shift",
                "temporal_patterns",
                "data_quality",
            },
            section="diagnostics",
        )

        return Phase3DiagnosticOptions(
            skewness=SkewnessDiagnosticOptions(
                **cls._mapping(
                    diagnostic_values.get(
                        "skewness",
                        {},
                    ),
                    section=(
                        "diagnostics.skewness"
                    ),
                )
            ),
            rare_categories=(
                RareCategoryDiagnosticOptions(
                    **cls._mapping(
                        diagnostic_values.get(
                            "rare_categories",
                            {},
                        ),
                        section=(
                            "diagnostics.rare_categories"
                        ),
                    )
                )
            ),
            high_cardinality=(
                HighCardinalityDiagnosticOptions(
                    **cls._mapping(
                        diagnostic_values.get(
                            "high_cardinality",
                            {},
                        ),
                        section=(
                            "diagnostics.high_cardinality"
                        ),
                    )
                )
            ),
            distribution_shift=DistributionShiftOptions(
                **cls._mapping(
                    diagnostic_values.get(
                        "distribution_shift",
                        {},
                    ),
                    section=(
                        "diagnostics.distribution_shift"
                    ),
                )
            ),
            temporal_patterns=TemporalPatternOptions(
                **cls._mapping(
                    diagnostic_values.get(
                        "temporal_patterns",
                        {},
                    ),
                    section=(
                        "diagnostics.temporal_patterns"
                    ),
                )
            ),
            data_quality=DataQualityDiagnosticOptions(
                **cls._mapping(
                    diagnostic_values.get(
                        "data_quality",
                        {},
                    ),
                    section=(
                        "diagnostics.data_quality"
                    ),
                )
            ),
        )

    @classmethod
    def _build_visualizations(
        cls,
        values: object,
    ) -> PlotManagerOptions:
        """Build visualization options."""

        visualization_values = cls._mapping(
            values,
            section="visualizations",
        )

        numeric_values = cls._mapping(
            visualization_values.pop(
                "numeric",
                {},
            ),
            section="visualizations.numeric",
        )

        categorical_values = cls._mapping(
            visualization_values.pop(
                "categorical",
                {},
            ),
            section=(
                "visualizations.categorical"
            ),
        )

        target_values = cls._mapping(
            visualization_values.pop(
                "target",
                {},
            ),
            section="visualizations.target",
        )

        missingness_values = cls._mapping(
            visualization_values.pop(
                "missingness",
                {},
            ),
            section=(
                "visualizations.missingness"
            ),
        )

        relationship_values = cls._mapping(
            visualization_values.pop(
                "relationships",
                {},
            ),
            section=(
                "visualizations.relationships"
            ),
        )

        return PlotManagerOptions(
            numeric=NumericPlotOptions(
                **numeric_values
            ),
            categorical=CategoricalPlotOptions(
                **categorical_values
            ),
            target=TargetPlotOptions(
                **target_values
            ),
            missingness=MissingnessPlotOptions(
                **missingness_values
            ),
            relationships=RelationshipPlotOptions(
                **relationship_values
            ),
            **visualization_values,
        )

    @classmethod
    def _build_recommendations(
        cls,
        values: object,
    ) -> RecommendationEngineOptions:
        """Build recommendation-engine options."""

        recommendation_values = cls._mapping(
            values,
            section="recommendations",
        )

        transformation_values = cls._mapping(
            recommendation_values.pop(
                "transformations",
                {},
            ),
            section=(
                "recommendations.transformations"
            ),
        )

        imputation_values = cls._mapping(
            recommendation_values.pop(
                "imputation",
                {},
            ),
            section=(
                "recommendations.imputation"
            ),
        )

        outlier_values = cls._mapping(
            recommendation_values.pop(
                "outliers",
                {},
            ),
            section=(
                "recommendations.outliers"
            ),
        )

        encoding_values = cls._mapping(
            recommendation_values.pop(
                "encoding",
                {},
            ),
            section=(
                "recommendations.encoding"
            ),
        )

        interaction_values = cls._mapping(
            recommendation_values.pop(
                "interactions",
                {},
            ),
            section=(
                "recommendations.interactions"
            ),
        )

        return RecommendationEngineOptions(
            transformations=(
                TransformationRecommendationOptions(
                    **transformation_values
                )
            ),
            imputation=ImputationRecommendationOptions(
                **imputation_values
            ),
            outliers=OutlierRecommendationOptions(
                **outlier_values
            ),
            encoding=EncodingRecommendationOptions(
                **encoding_values
            ),
            interactions=(
                InteractionRecommendationOptions(
                    **interaction_values
                )
            ),
            **recommendation_values,
        )

    @staticmethod
    def _mapping(
        value: object,
        *,
        section: str,
    ) -> dict[str, Any]:
        """Validate and copy one configuration mapping."""

        if value is None:
            return {}

        if not isinstance(value, Mapping):
            raise TypeError(
                f"Configuration section {section!r} "
                "must be a mapping."
            )

        return deepcopy(
            dict(value)
        )

    @staticmethod
    def _reject_unknown_keys(
        *,
        values: Mapping[str, Any],
        allowed: set[str],
        section: str,
    ) -> None:
        """Reject unknown nested configuration keys."""

        unknown_keys = set(
            values
        ).difference(
            allowed
        )

        if not unknown_keys:
            return

        unknown_text = ", ".join(
            sorted(unknown_keys)
        )

        raise ValueError(
            f"Unknown keys in {section!r}: "
            f"{unknown_text}"
        )

    @staticmethod
    def _load_yaml(
        path: Path,
    ) -> dict[str, Any]:
        """Load YAML configuration values."""

        try:
            values = yaml.safe_load(
                path.read_text(
                    encoding="utf-8"
                )
            )

        except yaml.YAMLError as error:
            raise ValueError(
                f"Invalid YAML configuration: {error}"
            ) from error

        return Phase3ConfigLoader._validate_root(
            values=values,
            path=path,
        )

    @staticmethod
    def _load_json(
        path: Path,
    ) -> dict[str, Any]:
        """Load JSON configuration values."""

        try:
            values = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )

        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON configuration: {error}"
            ) from error

        return Phase3ConfigLoader._validate_root(
            values=values,
            path=path,
        )

    @classmethod
    def _validate_input_path(
        cls,
        path: str | Path,
    ) -> Path:
        """Validate a configuration input path."""

        input_path = Path(path)

        if not input_path.exists():
            raise FileNotFoundError(
                f"Configuration file does not exist: "
                f"{input_path}"
            )

        if not input_path.is_file():
            raise IsADirectoryError(
                f"Configuration path is not a file: "
                f"{input_path}"
            )

        extension = input_path.suffix.casefold()

        if extension not in cls.supported_extensions:
            supported = ", ".join(
                sorted(cls.supported_extensions)
            )

            raise ValueError(
                f"Unsupported configuration extension "
                f"{extension!r}. Supported: {supported}"
            )

        return input_path

    @staticmethod
    def _validate_root(
        *,
        values: object,
        path: Path,
    ) -> dict[str, Any]:
        """Validate the root configuration mapping."""

        if values is None:
            raise ValueError(
                f"Configuration file is empty: {path}"
            )

        if not isinstance(values, dict):
            raise TypeError(
                "The configuration root must be a mapping."
            )

        if not all(
            isinstance(key, str)
            for key in values
        ):
            raise TypeError(
                "Every root configuration key must be "
                "a string."
            )

        return values

    @staticmethod
    def _format_from_extension(
        path: Path,
    ) -> ConfigFormat:
        """Resolve configuration format from file extension."""

        extension = path.suffix.casefold()

        if extension in {
            ".yaml",
            ".yml",
        }:
            return "yaml"

        if extension == ".json":
            return "json"

        raise ValueError(
            "Cannot determine configuration format from "
            f"{extension!r}."
        )

    @staticmethod
    def _set_dotted_value(
        *,
        values: dict[str, Any],
        dotted_path: str,
        value: Any,
    ) -> None:
        """Set one nested value using a dotted path."""

        if not dotted_path.strip():
            raise ValueError(
                "Override path cannot be empty."
            )

        path_parts = dotted_path.split(".")

        if any(
            not part.strip()
            for part in path_parts
        ):
            raise ValueError(
                f"Invalid dotted path: {dotted_path!r}"
            )

        current = values

        for path_part in path_parts[:-1]:
            existing = current.get(
                path_part
            )

            if existing is None:
                created: dict[str, Any] = {}

                current[path_part] = created
                current = created

                continue

            if not isinstance(existing, dict):
                raise TypeError(
                    f"Cannot set {dotted_path!r} because "
                    f"{path_part!r} is not a mapping."
                )

            current = existing

        current[path_parts[-1]] = deepcopy(
            value
        )
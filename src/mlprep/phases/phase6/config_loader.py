"""Configuration loading utilities for Phase 6."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, ClassVar, Literal

import yaml

from mlprep.phases.phase6.categorical import (
    CategoricalPipelineOptions,
    FrequencyEncoderOptions,
    HashingEncoderOptions,
    OneHotEncoderOptions,
    OrdinalEncoderOptions,
    RareCategoryGrouperOptions,
)
from mlprep.phases.phase6.config import (
    Phase6Config,
    Phase6ExecutionOptions,
    Phase6ValidationOptions,
)
from mlprep.phases.phase6.contract import (
    CategoricalEncodingKind,
    NumericScalingKind,
    NumericTransformationKind,
    SourceColumnPolicy,
    TransformationContractOptions,
    TransformationPlanOptions,
)
from mlprep.phases.phase6.feature_engineering import (
    BooleanFeatureOptions,
    DatetimeFeatureOptions,
    TextFeatureOptions,
)
from mlprep.phases.phase6.numeric import (
    BoxCoxTransformerOptions,
    LogTransformerOptions,
    MinMaxScalerOptions,
    NumericPipelineOptions,
    PowerTransformerOptions,
    RobustScalerOptions,
    StandardScalerOptions,
)
from mlprep.phases.phase6.orchestration import (
    PartitionTransformerOptions,
    TransformationPipelineOptions,
)
from mlprep.phases.phase6.output_governance import (
    Phase6ReadinessOptions,
    Phase6ReportOptions,
)
from mlprep.phases.phase6.validation import (
    NumericIntegrityOptions,
    RowIntegrityOptions,
    SchemaConsistencyOptions,
    TargetIntegrityOptions,
    UnknownCategoryOptions,
)

ConfigFormat = Literal[
    "yaml",
    "json",
]


class Phase6ConfigLoader:
    """Load and save Phase 6 configuration."""

    allowed_sections: ClassVar[frozenset[str]] = frozenset({
        "execution",
        "contract",
        "plan",
        "numeric",
        "categorical",
        "boolean",
        "datetime",
        "text",
        "partition_transformer",
        "pipeline",
        "validation",
        "readiness",
        "reporting",
    })

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        overrides: Mapping[str, Any] | None = None,
        dotted_overrides: Mapping[str, Any] | None = None,
    ) -> Phase6Config:
        """Load a YAML or JSON configuration."""

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

    @staticmethod
    def load_raw(
        path: str | Path,
    ) -> dict[str, Any]:
        """Load raw configuration values."""

        input_path = Path(
            path
        )

        if not input_path.exists():
            raise FileNotFoundError(
                f"Configuration does not exist: {input_path}"
            )

        if not input_path.is_file():
            raise IsADirectoryError(
                f"Configuration is not a file: {input_path}"
            )

        suffix = input_path.suffix.casefold()

        content = input_path.read_text(
            encoding="utf-8"
        )

        if suffix in {
            ".yaml",
            ".yml",
        }:
            values = yaml.safe_load(
                content
            )

        elif suffix == ".json":
            values = json.loads(
                content
            )

        else:
            raise ValueError(
                f"Unsupported configuration extension: {suffix!r}"
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
    ) -> Phase6Config:
        """Build configuration from a mapping."""

        root = deepcopy(
            dict(values)
        )

        unknown = set(
            root
        ).difference(
            cls.allowed_sections
        )

        if unknown:
            raise ValueError(
                "Unknown Phase 6 sections: "
                f"{sorted(unknown)}"
            )

        pipeline_values = cls._mapping(
            root.get(
                "pipeline",
                {},
            ),
            "pipeline",
        )
        for section in (
            "numeric",
            "categorical",
            "boolean",
            "datetime",
            "text",
            "partition_transformer",
        ):
            if section not in pipeline_values:
                continue
            nested_value = pipeline_values.pop(section)
            if section in root and root[section] != nested_value:
                raise ValueError(
                    f"Conflicting Phase 6 {section} configuration "
                    "between the root and pipeline sections."
                )
            root.setdefault(section, nested_value)

        numeric = cls._numeric(
            root.get(
                "numeric",
                {},
            )
        )

        categorical = cls._categorical(
            root.get(
                "categorical",
                {},
            )
        )

        boolean = BooleanFeatureOptions(
            **cls._mapping(
                root.get(
                    "boolean",
                    {},
                ),
                "boolean",
            )
        )

        datetime = DatetimeFeatureOptions(
            **cls._mapping(
                root.get(
                    "datetime",
                    {},
                ),
                "datetime",
            )
        )

        text = TextFeatureOptions(
            **cls._mapping(
                root.get(
                    "text",
                    {},
                ),
                "text",
            )
        )

        transformer = PartitionTransformerOptions(
            **cls._mapping(
                root.get(
                    "partition_transformer",
                    {},
                ),
                "partition_transformer",
            )
        )

        pipeline = TransformationPipelineOptions(
            numeric=numeric,
            categorical=categorical,
            boolean=boolean,
            datetime=datetime,
            text=text,
            partition_transformer=transformer,
            **pipeline_values,
        )

        return Phase6Config(
            execution=Phase6ExecutionOptions(
                **cls._mapping(
                    root.get(
                        "execution",
                        {},
                    ),
                    "execution",
                )
            ),
            contract=TransformationContractOptions(
                **cls._mapping(
                    root.get(
                        "contract",
                        {},
                    ),
                    "contract",
                )
            ),
            plan=cls._plan(
                root.get(
                    "plan",
                    {},
                )
            ),
            numeric=numeric,
            categorical=categorical,
            boolean=boolean,
            datetime=datetime,
            text=text,
            partition_transformer=transformer,
            pipeline=pipeline,
            validation=cls._validation(
                root.get(
                    "validation",
                    {},
                )
            ),
            readiness=Phase6ReadinessOptions(
                **cls._mapping(
                    root.get(
                        "readiness",
                        {},
                    ),
                    "readiness",
                )
            ),
            reporting=Phase6ReportOptions(
                **cls._mapping(
                    root.get(
                        "reporting",
                        {},
                    ),
                    "reporting",
                )
            ),
        )

    @classmethod
    def _plan(
        cls,
        values: object,
    ) -> TransformationPlanOptions:
        """Build plan options."""

        data = cls._mapping(
            values,
            "plan",
        )

        enum_fields = {
            "default_numeric_transformation": (
                NumericTransformationKind
            ),
            "default_numeric_scaling": (
                NumericScalingKind
            ),
            "default_categorical_encoding": (
                CategoricalEncodingKind
            ),
            "default_source_column_policy": (
                SourceColumnPolicy
            ),
        }

        for field_name, enum_type in enum_fields.items():
            if field_name in data:
                data[
                    field_name
                ] = enum_type(
                    data[field_name]
                )

        override_fields = {
            "numeric_transformation_overrides": (
                NumericTransformationKind
            ),
            "numeric_scaling_overrides": (
                NumericScalingKind
            ),
            "categorical_encoding_overrides": (
                CategoricalEncodingKind
            ),
            "source_column_policy_overrides": (
                SourceColumnPolicy
            ),
        }

        for field_name, enum_type in override_fields.items():
            if field_name not in data:
                continue

            mapping = cls._mapping(
                data[field_name],
                f"plan.{field_name}",
            )

            data[field_name] = {
                str(column): enum_type(
                    value
                )
                for column, value
                in mapping.items()
            }

        return TransformationPlanOptions(
            **data
        )

    @classmethod
    def _numeric(
        cls,
        values: object,
    ) -> NumericPipelineOptions:
        """Build numeric pipeline options."""

        data = cls._mapping(
            values,
            "numeric",
        )

        return NumericPipelineOptions(
            log=LogTransformerOptions(
                **cls._mapping(
                    data.pop(
                        "log",
                        {},
                    ),
                    "numeric.log",
                )
            ),
            power=PowerTransformerOptions(
                **cls._mapping(
                    data.pop(
                        "power",
                        {},
                    ),
                    "numeric.power",
                )
            ),
            box_cox=BoxCoxTransformerOptions(
                **cls._mapping(
                    data.pop(
                        "box_cox",
                        {},
                    ),
                    "numeric.box_cox",
                )
            ),
            standard=StandardScalerOptions(
                **cls._mapping(
                    data.pop(
                        "standard",
                        {},
                    ),
                    "numeric.standard",
                )
            ),
            robust=RobustScalerOptions(
                **cls._mapping(
                    data.pop(
                        "robust",
                        {},
                    ),
                    "numeric.robust",
                )
            ),
            minmax=MinMaxScalerOptions(
                **cls._mapping(
                    data.pop(
                        "minmax",
                        {},
                    ),
                    "numeric.minmax",
                )
            ),
            **data,
        )

    @classmethod
    def _categorical(
        cls,
        values: object,
    ) -> CategoricalPipelineOptions:
        """Build categorical pipeline options."""

        data = cls._mapping(
            values,
            "categorical",
        )

        return CategoricalPipelineOptions(
            rare_category=RareCategoryGrouperOptions(
                **cls._mapping(
                    data.pop(
                        "rare_category",
                        {},
                    ),
                    "categorical.rare_category",
                )
            ),
            one_hot=OneHotEncoderOptions(
                **cls._mapping(
                    data.pop(
                        "one_hot",
                        {},
                    ),
                    "categorical.one_hot",
                )
            ),
            ordinal=OrdinalEncoderOptions(
                **cls._mapping(
                    data.pop(
                        "ordinal",
                        {},
                    ),
                    "categorical.ordinal",
                )
            ),
            frequency=FrequencyEncoderOptions(
                **cls._mapping(
                    data.pop(
                        "frequency",
                        {},
                    ),
                    "categorical.frequency",
                )
            ),
            hashing=HashingEncoderOptions(
                **cls._mapping(
                    data.pop(
                        "hashing",
                        {},
                    ),
                    "categorical.hashing",
                )
            ),
            **data,
        )

    @classmethod
    def _validation(
        cls,
        values: object,
    ) -> Phase6ValidationOptions:
        """Build validation options."""

        data = cls._mapping(
            values,
            "validation",
        )

        allowed = {
            "schema_consistency",
            "numeric_integrity",
            "unknown_categories",
            "target_integrity",
            "row_integrity",
        }

        unknown = set(
            data
        ).difference(
            allowed
        )

        if unknown:
            raise ValueError(
                "Unknown validation options: "
                f"{sorted(unknown)}"
            )

        return Phase6ValidationOptions(
            schema_consistency=SchemaConsistencyOptions(
                **cls._mapping(
                    data.get(
                        "schema_consistency",
                        {},
                    ),
                    "validation.schema_consistency",
                )
            ),
            numeric_integrity=NumericIntegrityOptions(
                **cls._mapping(
                    data.get(
                        "numeric_integrity",
                        {},
                    ),
                    "validation.numeric_integrity",
                )
            ),
            unknown_categories=UnknownCategoryOptions(
                **cls._mapping(
                    data.get(
                        "unknown_categories",
                        {},
                    ),
                    "validation.unknown_categories",
                )
            ),
            target_integrity=TargetIntegrityOptions(
                **cls._mapping(
                    data.get(
                        "target_integrity",
                        {},
                    ),
                    "validation.target_integrity",
                )
            ),
            row_integrity=RowIntegrityOptions(
                **cls._mapping(
                    data.get(
                        "row_integrity",
                        {},
                    ),
                    "validation.row_integrity",
                )
            ),
        )

    @staticmethod
    def save(
        config: Phase6Config,
        path: str | Path,
        *,
        overwrite: bool = False,
    ) -> Path:
        """Save configuration as YAML or JSON."""

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

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        values = config.to_dict()

        if output_path.suffix.casefold() in {
            ".yaml",
            ".yml",
        }:
            content = yaml.safe_dump(
                values,
                sort_keys=False,
                allow_unicode=True,
            )

        elif output_path.suffix.casefold() == ".json":
            content = json.dumps(
                values,
                indent=2,
                ensure_ascii=False,
                default=str,
            )

        else:
            raise ValueError(
                "Output path must use .yaml, .yml, or .json."
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
        """Recursively merge configuration values."""

        merged = deepcopy(
            dict(base)
        )

        for key, value in override.items():
            if (
                isinstance(
                    merged.get(key),
                    Mapping,
                )
                and isinstance(
                    value,
                    Mapping,
                )
            ):
                merged[key] = cls.merge(
                    base=merged[key],
                    override=value,
                )

            else:
                merged[key] = deepcopy(
                    value
                )

        return merged

    @classmethod
    def apply_dot_overrides(
        cls,
        *,
        values: Mapping[str, Any],
        overrides: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Apply dotted-path overrides."""

        updated = deepcopy(
            dict(values)
        )

        for path, value in overrides.items():
            parts = path.split(
                "."
            )

            if (
                not path.strip()
                or any(
                    not part.strip()
                    for part in parts
                )
            ):
                raise ValueError(
                    f"Invalid dotted path: {path!r}"
                )

            current = updated

            for part in parts[:-1]:
                existing = current.get(
                    part
                )

                if existing is None:
                    current[
                        part
                    ] = {}

                    existing = current[
                        part
                    ]

                if not isinstance(
                    existing,
                    dict,
                ):
                    raise TypeError(
                        f"{part!r} is not a mapping."
                    )

                current = existing

            current[
                parts[-1]
            ] = deepcopy(
                value
            )

        return updated

    @staticmethod
    def _mapping(
        value: object,
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
                f"Section {section!r} must be a mapping."
            )

        return deepcopy(
            dict(value)
        )

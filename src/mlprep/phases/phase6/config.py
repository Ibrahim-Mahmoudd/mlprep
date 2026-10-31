"""Unified configuration for Phase 6."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from mlprep.phases.phase6.categorical import (
    CategoricalPipelineOptions,
)
from mlprep.phases.phase6.contract import (
    TransformationContractOptions,
    TransformationPlanOptions,
)
from mlprep.phases.phase6.feature_engineering import (
    BooleanFeatureOptions,
    DatetimeFeatureOptions,
    TextFeatureOptions,
)
from mlprep.phases.phase6.numeric import (
    NumericPipelineOptions,
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


@dataclass(slots=True)
class Phase6ExecutionOptions:
    """General execution options for Phase 6."""

    enabled: bool = True

    stop_on_contract_failure: bool = True
    stop_on_plan_failure: bool = True
    stop_on_component_exception: bool = True

    require_authorized_test_partition: bool = True
    keep_test_partition_protected: bool = True

    store_phase_state_in_result: bool = True
    store_execution_metadata: bool = True

    def __post_init__(self) -> None:
        """Validate execution options."""

        for name in (
            "enabled",
            "stop_on_contract_failure",
            "stop_on_plan_failure",
            "stop_on_component_exception",
            "require_authorized_test_partition",
            "keep_test_partition_protected",
            "store_phase_state_in_result",
            "store_execution_metadata",
        ):
            value = getattr(
                self,
                name,
            )

            if not isinstance(
                value,
                bool,
            ):
                raise TypeError(
                    f"{name} must be a Boolean."
                )


@dataclass(slots=True)
class Phase6ValidationOptions:
    """Complete Phase 6 validation configuration."""

    schema_consistency: SchemaConsistencyOptions = field(
        default_factory=SchemaConsistencyOptions
    )

    numeric_integrity: NumericIntegrityOptions = field(
        default_factory=NumericIntegrityOptions
    )

    unknown_categories: UnknownCategoryOptions = field(
        default_factory=UnknownCategoryOptions
    )

    target_integrity: TargetIntegrityOptions = field(
        default_factory=TargetIntegrityOptions
    )

    row_integrity: RowIntegrityOptions = field(
        default_factory=RowIntegrityOptions
    )


@dataclass(slots=True)
class Phase6Config:
    """Complete configuration for Phase 6."""

    execution: Phase6ExecutionOptions = field(
        default_factory=Phase6ExecutionOptions
    )

    contract: TransformationContractOptions = field(
        default_factory=TransformationContractOptions
    )

    plan: TransformationPlanOptions = field(
        default_factory=TransformationPlanOptions
    )

    numeric: NumericPipelineOptions = field(
        default_factory=NumericPipelineOptions
    )

    categorical: CategoricalPipelineOptions = field(
        default_factory=CategoricalPipelineOptions
    )

    boolean: BooleanFeatureOptions = field(
        default_factory=BooleanFeatureOptions
    )

    datetime: DatetimeFeatureOptions = field(
        default_factory=DatetimeFeatureOptions
    )

    text: TextFeatureOptions = field(
        default_factory=TextFeatureOptions
    )

    partition_transformer: PartitionTransformerOptions = field(
        default_factory=PartitionTransformerOptions
    )

    pipeline: TransformationPipelineOptions = field(
        default_factory=TransformationPipelineOptions
    )

    validation: Phase6ValidationOptions = field(
        default_factory=Phase6ValidationOptions
    )

    readiness: Phase6ReadinessOptions = field(
        default_factory=Phase6ReadinessOptions
    )

    reporting: Phase6ReportOptions = field(
        default_factory=Phase6ReportOptions
    )

    def __post_init__(self) -> None:
        """Validate and synchronize configuration."""

        self._synchronize_pipeline()
        self._synchronize_test_governance()
        self._synchronize_output_policy()
        self._validate_plan_policy()

    def _synchronize_pipeline(
        self,
    ) -> None:
        """Synchronize child configuration objects."""

        self.pipeline.numeric = self.numeric

        self.pipeline.categorical = (
            self.categorical
        )

        self.pipeline.boolean = self.boolean
        self.pipeline.datetime = self.datetime
        self.pipeline.text = self.text

        self.pipeline.partition_transformer = (
            self.partition_transformer
        )

    def _synchronize_test_governance(
        self,
    ) -> None:
        """Synchronize test-governance settings."""

        self.pipeline.test_transform_authorized = (
            self.execution
            .require_authorized_test_partition
        )

        self.readiness.require_authorized_test_transform = (
            self.execution
            .require_authorized_test_partition
        )

        self.readiness.require_protected_test_partition = (
            self.execution
            .keep_test_partition_protected
        )

    def _synchronize_output_policy(
        self,
    ) -> None:
        """Synchronize transformed-output validation."""

        self.validation.numeric_integrity.allow_missing_values = (
            self.partition_transformer
            .allow_missing_output_values
        )

        self.validation.numeric_integrity.allow_nonfinite_values = (
            self.partition_transformer
            .allow_nonfinite_output_values
        )

        self.validation.schema_consistency.require_numeric_output = (
            self.partition_transformer
            .require_numeric_output
        )

    def _validate_plan_policy(
        self,
    ) -> None:
        """Validate transformation-plan policy."""

        ordinal_columns = {
            column
            for column, encoder
            in self.plan
            .categorical_encoding_overrides
            .items()
            if encoder.value == "ordinal"
        }

        missing_orders = sorted(
            ordinal_columns.difference(
                self.plan.ordinal_orders
            )
        )

        if missing_orders:
            raise ValueError(
                "Ordinal encoding requires explicit category "
                f"orders for: {missing_orders}"
            )

        hashing_columns = {
            column
            for column, encoder
            in self.plan
            .categorical_encoding_overrides
            .items()
            if encoder.value == "hashing"
        }

        if (
            hashing_columns
            and not self.plan.allow_hashing_encoding
        ):
            raise ValueError(
                "Hashing overrides are configured while "
                "hashing encoding is disabled."
            )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """Convert configuration to a dictionary."""

        return self._serialize(
            asdict(self)
        )

    @classmethod
    def _serialize(
        cls,
        value: Any,
    ) -> Any:
        """Recursively serialize configuration values."""

        if isinstance(
            value,
            dict,
        ):
            return {
                str(key): cls._serialize(
                    item
                )
                for key, item
                in value.items()
            }

        if isinstance(
            value,
            (list, tuple, set),
        ):
            return [
                cls._serialize(item)
                for item in value
            ]

        if hasattr(
            value,
            "model_dump",
        ):
            return cls._serialize(
                value.model_dump(
                    mode="json"
                )
            )

        if hasattr(
            value,
            "value",
        ):
            return value.value

        return value
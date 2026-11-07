"""Configuration loading utilities for Phase 2."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, ClassVar, Literal

import yaml

from mlprep.phases.phase2.config import (
    Phase2Config,
    Phase2ExecutionOptions,
    Phase2StrategyOptions,
    Phase2ValidationOptions,
)
from mlprep.phases.phase2.contract.split_contract import (
    SplitContractOptions,
)
from mlprep.phases.phase2.contract.split_roles import (
    SplitRoleOptions,
)
from mlprep.phases.phase2.leakage_firewall.duplicate_overlap import (
    DuplicateOverlapOptions,
)
from mlprep.phases.phase2.leakage_firewall.firewall import (
    LeakageFirewallOptions,
)
from mlprep.phases.phase2.leakage_firewall.group_overlap import (
    GroupOverlapOptions,
)
from mlprep.phases.phase2.leakage_firewall.post_outcome import (
    PostOutcomeLeakageOptions,
)
from mlprep.phases.phase2.leakage_firewall.preprocessing_leakage import (
    PreprocessingLeakageOptions,
)
from mlprep.phases.phase2.leakage_firewall.row_overlap import (
    RowOverlapOptions,
)
from mlprep.phases.phase2.leakage_firewall.target_leakage import (
    TargetLeakageOptions,
)
from mlprep.phases.phase2.leakage_firewall.temporal_leakage import (
    TemporalLeakageOptions,
)
from mlprep.phases.phase2.output_governance.fingerprint import (
    Phase2FingerprintOptions,
)
from mlprep.phases.phase2.output_governance.metadata import (
    SplitRatios,
    SplitStrategyName,
)
from mlprep.phases.phase2.output_governance.readiness import (
    Phase2ReadinessOptions,
)
from mlprep.phases.phase2.output_governance.report import (
    Phase2ReportOptions,
)
from mlprep.phases.phase2.output_governance.test_lock import (
    TestLockOptions,
)
from mlprep.phases.phase2.strategies.group_split import (
    GroupSplitOptions,
)
from mlprep.phases.phase2.strategies.group_stratified_split import (
    GroupStratifiedSplitOptions,
)
from mlprep.phases.phase2.strategies.predefined_split import (
    PredefinedSplitOptions,
)
from mlprep.phases.phase2.strategies.purged_time_split import (
    PurgedTimeSplitOptions,
)
from mlprep.phases.phase2.strategies.random_split import (
    RandomSplitOptions,
)
from mlprep.phases.phase2.strategies.regression_stratified_split import (
    RegressionStratifiedSplitOptions,
)
from mlprep.phases.phase2.strategies.stratified_split import (
    StratifiedSplitOptions,
)
from mlprep.phases.phase2.strategies.time_split import (
    TimeSplitOptions,
)
from mlprep.phases.phase2.validation.class_distribution import (
    ClassDistributionValidationOptions,
)
from mlprep.phases.phase2.validation.group_distribution import (
    GroupDistributionValidationOptions,
)
from mlprep.phases.phase2.validation.partition_integrity import (
    PartitionIntegrityValidationOptions,
)
from mlprep.phases.phase2.validation.split_sizes import (
    SplitSizeValidationOptions,
)
from mlprep.phases.phase2.validation.target_distribution import (
    TargetDistributionValidationOptions,
)
from mlprep.phases.phase2.validation.temporal_boundaries import (
    TemporalBoundaryValidationOptions,
)

ConfigFormat = Literal[
    "yaml",
    "json",
]


class Phase2ConfigLoader:
    """Load, merge, validate, and save Phase 2 configurations."""

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
        "strategy",
        "ratios",
        "random_seed",
        "execution",
        "split_roles",
        "contract",
        "strategies",
        "leakage_firewall",
        "validation",
        "fingerprint",
        "test_lock",
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
    ) -> Phase2Config:
        """Load a Phase 2 configuration from YAML or JSON."""

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
            return cls._load_yaml(input_path)

        if extension in cls.json_extensions:
            return cls._load_json(input_path)

        supported = ", ".join(
            sorted(cls.supported_extensions)
        )

        raise ValueError(
            f"Unsupported configuration extension "
            f"{extension!r}. Supported extensions: "
            f"{supported}"
        )

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Any],
    ) -> Phase2Config:
        """Create a Phase 2 configuration from a mapping."""

        normalized_values = deepcopy(
            dict(values)
        )

        unknown_sections = set(
            normalized_values
        ).difference(
            cls.allowed_sections
        )

        if unknown_sections:
            sections = ", ".join(
                sorted(unknown_sections)
            )

            raise ValueError(
                "Unknown Phase 2 configuration sections: "
                f"{sections}"
            )

        if "strategy" not in normalized_values:
            raise ValueError(
                "The Phase 2 configuration requires "
                "a strategy."
            )

        strategy = SplitStrategyName(
            normalized_values["strategy"]
        )

        ratios = cls._build_ratios(
            normalized_values.get(
                "ratios",
                {},
            )
        )

        random_seed = int(
            normalized_values.get(
                "random_seed",
                42,
            )
        )

        execution = Phase2ExecutionOptions(
            **cls._mapping(
                normalized_values.get(
                    "execution",
                    {},
                ),
                section="execution",
            )
        )

        split_roles = SplitRoleOptions(
            **cls._mapping(
                normalized_values.get(
                    "split_roles",
                    {},
                ),
                section="split_roles",
            )
        )

        strategies = cls._build_strategies(
            values=normalized_values.get(
                "strategies",
                {},
            ),
            ratios=ratios,
            random_seed=random_seed,
        )

        contract_values = normalized_values.get(
            "contract"
        )

        contract = None

        if contract_values is not None:
            resolved_contract_values = (
                cls._mapping(
                    contract_values,
                    section="contract",
                )
            )

            resolved_contract_values[
                "strategy"
            ] = SplitStrategyName(
                resolved_contract_values.get(
                    "strategy",
                    strategy,
                )
            )

            resolved_contract_values[
                "ratios"
            ] = cls._build_ratios(
                resolved_contract_values.get(
                    "ratios",
                    ratios.model_dump(),
                )
            )

            contract = SplitContractOptions(
                **resolved_contract_values
            )

        leakage_firewall = (
            cls._build_leakage_firewall(
                normalized_values.get(
                    "leakage_firewall",
                    {},
                )
            )
        )

        validation = cls._build_validation(
            normalized_values.get(
                "validation",
                {},
            ),
            ratios=ratios,
        )

        fingerprint = Phase2FingerprintOptions(
            **cls._mapping(
                normalized_values.get(
                    "fingerprint",
                    {},
                ),
                section="fingerprint",
            )
        )

        test_lock = TestLockOptions(
            **cls._mapping(
                normalized_values.get(
                    "test_lock",
                    {},
                ),
                section="test_lock",
            )
        )

        readiness = Phase2ReadinessOptions(
            **cls._mapping(
                normalized_values.get(
                    "readiness",
                    {},
                ),
                section="readiness",
            )
        )

        reporting = Phase2ReportOptions(
            **cls._mapping(
                normalized_values.get(
                    "reporting",
                    {},
                ),
                section="reporting",
            )
        )

        return Phase2Config(
            strategy=strategy,
            ratios=ratios,
            random_seed=random_seed,
            execution=execution,
            split_roles=split_roles,
            contract=contract,
            strategies=strategies,
            leakage_firewall=leakage_firewall,
            validation=validation,
            fingerprint=fingerprint,
            test_lock=test_lock,
            readiness=readiness,
            reporting=reporting,
        )

    @classmethod
    def save(
        cls,
        config: Phase2Config,
        path: str | Path,
        *,
        config_format: ConfigFormat | None = None,
        overwrite: bool = False,
    ) -> Path:
        """Save a resolved Phase 2 configuration."""

        if not isinstance(config, Phase2Config):
            raise TypeError(
                "config must be an instance of Phase2Config."
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
            errors.append(str(error))

        return errors

    @classmethod
    def merge(
        cls,
        *,
        base: Mapping[str, Any],
        override: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Recursively merge two mappings."""

        merged = deepcopy(
            dict(base)
        )

        for key, override_value in override.items():
            base_value = merged.get(key)

            if (
                isinstance(
                    base_value,
                    Mapping,
                )
                and isinstance(
                    override_value,
                    Mapping,
                )
            ):
                merged[key] = cls.merge(
                    base=base_value,
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
        """Apply nested overrides using dotted paths."""

        updated_values = deepcopy(
            dict(values)
        )

        for dotted_path, value in overrides.items():
            cls._set_dotted_value(
                values=updated_values,
                dotted_path=dotted_path,
                value=value,
            )

        return updated_values

    @classmethod
    def _build_strategies(
        cls,
        *,
        values: object,
        ratios: SplitRatios,
        random_seed: int,
    ) -> Phase2StrategyOptions:
        """Build every split-strategy option model."""

        strategy_values = cls._mapping(
            values,
            section="strategies",
        )

        random_values = cls._mapping(
            strategy_values.get(
                "random",
                {},
            ),
            section="strategies.random",
        )

        stratified_values = cls._mapping(
            strategy_values.get(
                "stratified",
                {},
            ),
            section="strategies.stratified",
        )

        group_values = cls._mapping(
            strategy_values.get(
                "group",
                {},
            ),
            section="strategies.group",
        )

        group_stratified_values = (
            cls._mapping(
                strategy_values.get(
                    "group_stratified",
                    {},
                ),
                section=(
                    "strategies.group_stratified"
                ),
            )
        )

        time_values = cls._mapping(
            strategy_values.get(
                "time",
                {},
            ),
            section="strategies.time",
        )

        purged_time_values = cls._mapping(
            strategy_values.get(
                "purged_time",
                {},
            ),
            section="strategies.purged_time",
        )

        regression_stratified_values = (
            cls._mapping(
                strategy_values.get(
                    "regression_stratified",
                    {},
                ),
                section=(
                    "strategies."
                    "regression_stratified"
                ),
            )
        )

        predefined_values = strategy_values.get(
            "predefined"
        )

        predefined = None

        if predefined_values is not None:
            predefined = PredefinedSplitOptions(
                **cls._mapping(
                    predefined_values,
                    section=(
                        "strategies.predefined"
                    ),
                )
            )

        return Phase2StrategyOptions(
            random=RandomSplitOptions(
                ratios=cls._pop_ratios(
                    random_values,
                    ratios,
                ),
                random_seed=int(
                    random_values.pop(
                        "random_seed",
                        random_seed,
                    )
                ),
                **random_values,
            ),
            stratified=StratifiedSplitOptions(
                ratios=cls._pop_ratios(
                    stratified_values,
                    ratios,
                ),
                random_seed=int(
                    stratified_values.pop(
                        "random_seed",
                        random_seed,
                    )
                ),
                **stratified_values,
            ),
            group=GroupSplitOptions(
                ratios=cls._pop_ratios(
                    group_values,
                    ratios,
                ),
                random_seed=int(
                    group_values.pop(
                        "random_seed",
                        random_seed,
                    )
                ),
                **group_values,
            ),
            group_stratified=(
                GroupStratifiedSplitOptions(
                    ratios=cls._pop_ratios(
                        group_stratified_values,
                        ratios,
                    ),
                    random_seed=int(
                        group_stratified_values.pop(
                            "random_seed",
                            random_seed,
                        )
                    ),
                    **group_stratified_values,
                )
            ),
            time=TimeSplitOptions(
                ratios=cls._pop_ratios(
                    time_values,
                    ratios,
                ),
                **time_values,
            ),
            purged_time=PurgedTimeSplitOptions(
                ratios=cls._pop_ratios(
                    purged_time_values,
                    ratios,
                ),
                **purged_time_values,
            ),
            regression_stratified=(
                RegressionStratifiedSplitOptions(
                    ratios=cls._pop_ratios(
                        regression_stratified_values,
                        ratios,
                    ),
                    random_seed=int(
                        regression_stratified_values.pop(
                            "random_seed",
                            random_seed,
                        )
                    ),
                    **regression_stratified_values,
                )
            ),
            predefined=predefined,
        )

    @classmethod
    def _build_leakage_firewall(
        cls,
        values: object,
    ) -> LeakageFirewallOptions:
        """Build leakage-firewall options."""

        firewall_values = cls._mapping(
            values,
            section="leakage_firewall",
        )

        target_values = cls._mapping(
            firewall_values.pop(
                "target_leakage",
                {},
            ),
            section=(
                "leakage_firewall.target_leakage"
            ),
        )

        row_values = cls._mapping(
            firewall_values.pop(
                "row_overlap",
                {},
            ),
            section=(
                "leakage_firewall.row_overlap"
            ),
        )

        group_values = cls._mapping(
            firewall_values.pop(
                "group_overlap",
                {},
            ),
            section=(
                "leakage_firewall.group_overlap"
            ),
        )

        duplicate_values = cls._mapping(
            firewall_values.pop(
                "duplicate_overlap",
                {},
            ),
            section=(
                "leakage_firewall."
                "duplicate_overlap"
            ),
        )

        temporal_values = cls._mapping(
            firewall_values.pop(
                "temporal_leakage",
                {},
            ),
            section=(
                "leakage_firewall."
                "temporal_leakage"
            ),
        )

        post_outcome_values = cls._mapping(
            firewall_values.pop(
                "post_outcome",
                {},
            ),
            section=(
                "leakage_firewall.post_outcome"
            ),
        )

        preprocessing_values = cls._mapping(
            firewall_values.pop(
                "preprocessing_leakage",
                {},
            ),
            section=(
                "leakage_firewall."
                "preprocessing_leakage"
            ),
        )

        return LeakageFirewallOptions(
            target_leakage=(
                TargetLeakageOptions(
                    **target_values
                )
            ),
            row_overlap=RowOverlapOptions(
                **row_values
            ),
            group_overlap=GroupOverlapOptions(
                **group_values
            ),
            duplicate_overlap=(
                DuplicateOverlapOptions(
                    **duplicate_values
                )
            ),
            temporal_leakage=(
                TemporalLeakageOptions(
                    **temporal_values
                )
            ),
            post_outcome=(
                PostOutcomeLeakageOptions(
                    **post_outcome_values
                )
            ),
            preprocessing_leakage=(
                PreprocessingLeakageOptions(
                    **preprocessing_values
                )
            ),
            **firewall_values,
        )

    @classmethod
    def _build_validation(
        cls,
        values: object,
        *,
        ratios: SplitRatios,
    ) -> Phase2ValidationOptions:
        """Build all validation options."""

        validation_values = cls._mapping(
            values,
            section="validation",
        )

        split_size_values = cls._mapping(
            validation_values.get(
                "split_sizes",
                {},
            ),
            section="validation.split_sizes",
        )

        requested_ratios = split_size_values.pop(
            "requested_ratios",
            None,
        )

        if requested_ratios is None:
            resolved_requested_ratios = (
                ratios.model_copy(
                    deep=True
                )
            )
        else:
            resolved_requested_ratios = (
                cls._build_ratios(
                    requested_ratios
                )
            )

        return Phase2ValidationOptions(
            split_sizes=(
                SplitSizeValidationOptions(
                    requested_ratios=(
                        resolved_requested_ratios
                    ),
                    **split_size_values,
                )
            ),
            class_distribution=(
                ClassDistributionValidationOptions(
                    **cls._mapping(
                        validation_values.get(
                            "class_distribution",
                            {},
                        ),
                        section=(
                            "validation."
                            "class_distribution"
                        ),
                    )
                )
            ),
            target_distribution=(
                TargetDistributionValidationOptions(
                    **cls._mapping(
                        validation_values.get(
                            "target_distribution",
                            {},
                        ),
                        section=(
                            "validation."
                            "target_distribution"
                        ),
                    )
                )
            ),
            group_distribution=(
                GroupDistributionValidationOptions(
                    **cls._mapping(
                        validation_values.get(
                            "group_distribution",
                            {},
                        ),
                        section=(
                            "validation."
                            "group_distribution"
                        ),
                    )
                )
            ),
            temporal_boundaries=(
                TemporalBoundaryValidationOptions(
                    **cls._mapping(
                        validation_values.get(
                            "temporal_boundaries",
                            {},
                        ),
                        section=(
                            "validation."
                            "temporal_boundaries"
                        ),
                    )
                )
            ),
            partition_integrity=(
                PartitionIntegrityValidationOptions(
                    **cls._mapping(
                        validation_values.get(
                            "partition_integrity",
                            {},
                        ),
                        section=(
                            "validation."
                            "partition_integrity"
                        ),
                    )
                )
            ),
        )

    @staticmethod
    def _build_ratios(
        values: object,
    ) -> SplitRatios:
        """Build partition-ratio options."""

        if isinstance(values, SplitRatios):
            return values.model_copy(
                deep=True
            )

        if not isinstance(values, Mapping):
            raise TypeError(
                "ratios must be a mapping."
            )

        return SplitRatios(
            **dict(values)
        )

    @classmethod
    def _pop_ratios(
        cls,
        values: dict[str, Any],
        default: SplitRatios,
    ) -> SplitRatios:
        """Pop strategy-specific ratios or copy defaults."""

        ratio_values = values.pop(
            "ratios",
            None,
        )

        if ratio_values is None:
            return default.model_copy(
                deep=True
            )

        return cls._build_ratios(
            ratio_values
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
                f"Invalid YAML configuration: "
                f"{error}"
            ) from error

        return Phase2ConfigLoader._validate_root(
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
                f"Invalid JSON configuration: "
                f"{error}"
            ) from error

        return Phase2ConfigLoader._validate_root(
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

        if (
            input_path.suffix.casefold()
            not in cls.supported_extensions
        ):
            supported = ", ".join(
                sorted(cls.supported_extensions)
            )

            raise ValueError(
                "Unsupported configuration extension "
                f"{input_path.suffix!r}. Supported: "
                f"{supported}"
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
                f"Configuration file is empty: "
                f"{path}"
            )

        if not isinstance(values, dict):
            raise TypeError(
                "The configuration root must be "
                "a mapping."
            )

        if not all(
            isinstance(key, str)
            for key in values
        ):
            raise TypeError(
                "Every root configuration key "
                "must be a string."
            )

        return values

    @staticmethod
    def _format_from_extension(
        path: Path,
    ) -> ConfigFormat:
        """Resolve format from a path extension."""

        extension = path.suffix.casefold()

        if extension in {
            ".yaml",
            ".yml",
        }:
            return "yaml"

        if extension == ".json":
            return "json"

        raise ValueError(
            "Cannot determine configuration "
            f"format from {extension!r}."
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
                f"Invalid dotted path: "
                f"{dotted_path!r}"
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
                    f"Cannot set {dotted_path!r} "
                    f"because {path_part!r} is not "
                    "a mapping."
                )

            current = existing

        current[path_parts[-1]] = deepcopy(
            value
        )
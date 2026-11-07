"""Required and protected feature validation for Phase 7."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase7.orchestration.selection_pipeline import (
    FeatureSelectionDecision,
)


class ProtectedFeatureIssue(BaseModel):
    """One protected-feature policy violation."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    column: str
    issue_type: str
    reason: str

    blocks_readiness: bool = True

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class ProtectedFeaturesResult(BaseModel):
    """Complete protected-feature validation output."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    valid: bool = True

    issues: list[
        ProtectedFeatureIssue
    ] = Field(
        default_factory=list,
    )

    retained_required_features: list[str] = Field(
        default_factory=list,
    )

    retained_protected_features: list[str] = Field(
        default_factory=list,
    )

    removed_prohibited_features: list[str] = Field(
        default_factory=list,
    )

    missing_required_features: list[str] = Field(
        default_factory=list,
    )

    missing_protected_features: list[str] = Field(
        default_factory=list,
    )

    present_prohibited_features: list[str] = Field(
        default_factory=list,
    )

    silently_restored_features: list[str] = Field(
        default_factory=list,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


@dataclass(slots=True)
class ProtectedFeaturesOptions:
    """Options controlling feature-policy validation."""

    enabled: bool = True

    require_all_required_features: bool = True
    require_all_protected_features: bool = True
    require_prohibited_features_absent: bool = True

    reject_silent_restoration: bool = True

    leakage_risk_features: list[str] = field(
        default_factory=list,
    )

    require_leakage_risk_features_absent: bool = True

    def __post_init__(self) -> None:
        """Validate protected-feature options."""

        normalized = [
            str(feature).strip()
            for feature
            in self.leakage_risk_features
        ]

        if any(
            not feature
            for feature in normalized
        ):
            raise ValueError(
                "leakage_risk_features cannot contain "
                "empty names."
            )

        if len(normalized) != len(
            set(normalized)
        ):
            raise ValueError(
                "leakage_risk_features contains duplicates."
            )

        self.leakage_risk_features = normalized


class ProtectedFeaturesValidator:
    """Validate required, protected, and prohibited features."""

    component_name = "protected_features"

    def __init__(
        self,
        options: ProtectedFeaturesOptions | None = None,
    ) -> None:
        """Initialize the validator."""

        self.options = (
            options
            or ProtectedFeaturesOptions()
        )

    def validate(
        self,
        *,
        selected_features: list[str],
        required_features: list[str],
        protected_features: list[str],
        prohibited_features: list[str],
        decisions: dict[
            str,
            FeatureSelectionDecision,
        ] | None = None,
    ) -> ProtectedFeaturesResult:
        """Validate final feature-policy compliance."""

        selected = self._normalize(
            selected_features,
            "selected_features",
        )

        required = self._normalize(
            required_features,
            "required_features",
        )

        protected = self._normalize(
            protected_features,
            "protected_features",
        )

        prohibited = self._normalize(
            prohibited_features,
            "prohibited_features",
        )

        if not self.options.enabled:
            return ProtectedFeaturesResult(
                metadata={
                    "enabled": False,
                },
            )

        selected_set = set(
            selected
        )

        missing_required = [
            feature
            for feature in required
            if feature not in selected_set
        ]

        missing_protected = [
            feature
            for feature in protected
            if feature not in selected_set
        ]

        present_prohibited = [
            feature
            for feature in prohibited
            if feature in selected_set
        ]

        leakage_risk_present = [
            feature
            for feature
            in self.options.leakage_risk_features
            if feature in selected_set
        ]

        issues: list[
            ProtectedFeatureIssue
        ] = []

        if (
            self.options.require_all_required_features
        ):
            for feature in missing_required:
                issues.append(
                    ProtectedFeatureIssue(
                        column=feature,
                        issue_type=(
                            "missing_required_feature"
                        ),
                        reason=(
                            "A required feature is absent "
                            "from the selected schema."
                        ),
                    )
                )

        if (
            self.options.require_all_protected_features
        ):
            for feature in missing_protected:
                issues.append(
                    ProtectedFeatureIssue(
                        column=feature,
                        issue_type=(
                            "missing_protected_feature"
                        ),
                        reason=(
                            "A protected feature is absent "
                            "from the selected schema."
                        ),
                    )
                )

        if (
            self.options
            .require_prohibited_features_absent
        ):
            for feature in present_prohibited:
                issues.append(
                    ProtectedFeatureIssue(
                        column=feature,
                        issue_type=(
                            "present_prohibited_feature"
                        ),
                        reason=(
                            "A prohibited feature remains "
                            "in the selected schema."
                        ),
                    )
                )

        if (
            self.options
            .require_leakage_risk_features_absent
        ):
            for feature in leakage_risk_present:
                issues.append(
                    ProtectedFeatureIssue(
                        column=feature,
                        issue_type=(
                            "leakage_risk_feature"
                        ),
                        reason=(
                            "A configured leakage-risk feature "
                            "remains selected."
                        ),
                    )
                )

        silently_restored: list[str] = []

        if decisions is not None:
            for feature, decision in decisions.items():
                if not decision.selected:
                    continue

                restored = bool(
                    decision.restored_for_minimum_count
                )

                documented = bool(
                    decision.required
                    or decision.protected
                    or (
                        "restored_for_feature_count_policy"
                        in decision.selection_reasons
                    )
                )

                if restored and not documented:
                    silently_restored.append(
                        feature
                    )

                    if self.options.reject_silent_restoration:
                        issues.append(
                            ProtectedFeatureIssue(
                                column=feature,
                                issue_type=(
                                    "silent_restoration"
                                ),
                                reason=(
                                    "The feature was restored "
                                    "without a documented policy."
                                ),
                            )
                        )

        return ProtectedFeaturesResult(
            valid=not any(
                issue.blocks_readiness
                for issue in issues
            ),
            issues=issues,
            retained_required_features=[
                feature
                for feature in required
                if feature in selected_set
            ],
            retained_protected_features=[
                feature
                for feature in protected
                if feature in selected_set
            ],
            removed_prohibited_features=[
                feature
                for feature in prohibited
                if feature not in selected_set
            ],
            missing_required_features=(
                missing_required
            ),
            missing_protected_features=(
                missing_protected
            ),
            present_prohibited_features=(
                present_prohibited
            ),
            silently_restored_features=(
                silently_restored
            ),
            metadata={
                "enabled": True,
                "leakage_risk_features_present": (
                    leakage_risk_present
                ),
            },
        )

    @staticmethod
    def _normalize(
        features: list[str],
        name: str,
    ) -> list[object]:
        """Normalize one feature-name list."""

        normalized = [
            str(feature).strip()
            for feature in features
        ]

        if any(
            not feature
            for feature in normalized
        ):
            raise ValueError(
                f"{name} cannot contain empty names."
            )

        if len(normalized) != len(
            set(normalized)
        ):
            raise ValueError(
                f"{name} contains duplicate names."
            )

        return normalized

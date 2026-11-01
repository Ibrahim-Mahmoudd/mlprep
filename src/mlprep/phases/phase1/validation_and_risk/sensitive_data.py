"""Potential sensitive-data detection for Phase 1."""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass, field

import pandas as pd
from pandas.api.types import (
    is_object_dtype,
    is_string_dtype,
)

from mlprep.phases.phase1.output_governance.result import (
    Finding,
    Phase1Result,
)
from mlprep.phases.phase1.output_governance.state import Phase1State
from mlprep.types import (
    FindingAction,
    FindingCategory,
    Severity,
)

_EMAIL_PATTERN = re.compile(
    r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
)

_PHONE_PATTERN = re.compile(
    r"^\+?[0-9][0-9\s().-]{6,20}$"
)

_URL_PATTERN = re.compile(
    r"^(?:https?://|www\.)\S+$",
    flags=re.IGNORECASE,
)

_PAYMENT_CARD_PATTERN = re.compile(
    r"^[0-9][0-9\s-]{11,24}$"
)


@dataclass(slots=True)
class SensitiveDataOptions:
    """Options controlling sensitive-data detection."""

    enabled: bool = True

    include_columns: list[str] | None = None
    exclude_columns: list[str] = field(default_factory=list)

    declared_sensitive_columns: list[str] = field(
        default_factory=list
    )

    detect_emails: bool = True
    detect_phone_numbers: bool = True
    detect_ip_addresses: bool = True
    detect_urls: bool = True
    detect_payment_cards: bool = True

    sample_size: int = 10_000
    detection_threshold: float = 0.80

    maximum_examples: int = 5

    def __post_init__(self) -> None:
        """Validate sensitive-data options."""

        if self.sample_size <= 0:
            raise ValueError(
                "sample_size must be positive."
            )

        if not 0.0 <= self.detection_threshold <= 1.0:
            raise ValueError(
                "detection_threshold must be between zero and one."
            )


class SensitiveDataDetector:
    """Detect columns containing potentially sensitive patterns."""

    component_name = "sensitive_data"

    def __init__(
        self,
        options: SensitiveDataOptions | None = None,
    ) -> None:
        """Initialize the detector."""

        self.options = options or SensitiveDataOptions()

    def inspect(
        self,
        data: pd.DataFrame,
        result: Phase1Result,
        state: Phase1State,
    ) -> dict[str, str]:
        """Inspect selected columns for sensitive patterns."""

        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "Sensitive-data detection requires a pandas DataFrame."
            )

        if not self.options.enabled:
            return {}

        detected: dict[str, str] = {}

        for column in self.options.declared_sensitive_columns:
            if column in data.columns:
                self._register_sensitive_column(
                    column=column,
                    sensitive_type="declared_sensitive",
                    match_ratio=1.0,
                    result=result,
                    state=state,
                )

                detected[column] = "declared_sensitive"

        for column in self._select_columns(data):
            series = (
                data[column]
                .dropna()
                .astype(str)
                .head(self.options.sample_size)
            )

            if series.empty:
                continue

            sensitive_type, match_ratio = (
                self._detect_pattern(series)
            )

            if sensitive_type is None:
                continue

            if (
                match_ratio
                < self.options.detection_threshold
            ):
                continue

            self._register_sensitive_column(
                column=column,
                sensitive_type=sensitive_type,
                match_ratio=match_ratio,
                result=result,
                state=state,
            )

            detected[column] = sensitive_type

        return detected

    def _detect_pattern(
        self,
        series: pd.Series,
    ) -> tuple[str | None, float]:
        """Return the strongest detected sensitive pattern."""

        detectors: list[
            tuple[str, callable]
        ] = []

        if self.options.detect_emails:
            detectors.append(
                (
                    "email",
                    lambda value: bool(
                        _EMAIL_PATTERN.fullmatch(value)
                    ),
                )
            )

        if self.options.detect_payment_cards:
            detectors.append(
                (
                    "payment_card",
                    self._looks_like_payment_card,
                )
            )

        if self.options.detect_phone_numbers:
            detectors.append(
                (
                    "phone_number",
                    lambda value: bool(
                        _PHONE_PATTERN.fullmatch(value)
                    ),
                )
            )

        if self.options.detect_urls:
            detectors.append(
                (
                    "url",
                    lambda value: bool(
                        _URL_PATTERN.fullmatch(value)
                    ),
                )
            )

        if self.options.detect_ip_addresses:
            detectors.append(
                (
                    "ip_address",
                    self._is_ip_address,
                )
            )

        best_type: str | None = None
        best_ratio = 0.0

        for sensitive_type, detector in detectors:
            match_count = sum(
                detector(value)
                for value in series
            )

            ratio = match_count / len(series)

            if ratio > best_ratio:
                best_type = sensitive_type
                best_ratio = ratio

        return best_type, best_ratio

    def _register_sensitive_column(
        self,
        *,
        column: str,
        sensitive_type: str,
        match_ratio: float,
        result: Phase1Result,
        state: Phase1State,
    ) -> None:
        """Mark one column as potentially sensitive."""

        column_state = state.column_states.get(column)

        if column_state is not None:
            column_state.is_sensitive = True
            column_state.sensitive_type = sensitive_type

            state.register_column_state(
                column_state
            )

        result.add_finding(
            Finding(
                code="P1-SENSITIVE-001",
                category=FindingCategory.SENSITIVE_DATA,
                severity=Severity.WARNING,
                component=self.component_name,
                message=(
                    f"Column {column!r} may contain sensitive "
                    f"data of type {sensitive_type!r}."
                ),
                columns=[column],
                recommended_action=FindingAction.WARN,
                applied_action=FindingAction.REPORT,
                blocks_readiness=False,
                details={
                    "sensitive_type": sensitive_type,
                    "match_ratio": match_ratio,
                },
            )
        )

    def _select_columns(
        self,
        data: pd.DataFrame,
    ) -> list[object]:
        """Select text-like columns for sensitive-data inspection."""

        excluded = set(
            self.options.exclude_columns
        )

        if self.options.include_columns is not None:
            return [
                column
                for column in self.options.include_columns
                if column in data.columns
                and column not in excluded
            ]

        return [
            str(column)
            for column in data.columns
            if str(column) not in excluded
            and (
                is_string_dtype(data[column].dtype)
                or is_object_dtype(data[column].dtype)
            )
        ]

    @staticmethod
    def _is_ip_address(
        value: str,
    ) -> bool:
        """Return whether a string is an IPv4 or IPv6 address."""

        try:
            ipaddress.ip_address(value.strip())
        except ValueError:
            return False

        return True

    @staticmethod
    def _looks_like_payment_card(
        value: str,
    ) -> bool:
        """Detect a card-like number using regex and Luhn checksum."""

        if not _PAYMENT_CARD_PATTERN.fullmatch(value):
            return False

        digits = [
            int(character)
            for character in value
            if character.isdigit()
        ]

        if not 12 <= len(digits) <= 19:
            return False

        checksum = 0
        parity = len(digits) % 2

        for index, digit in enumerate(digits):
            checked_digit = digit

            if index % 2 == parity:
                checked_digit *= 2

                if checked_digit > 9:
                    checked_digit -= 9

            checksum += checked_digit

        return checksum % 10 == 0

"""Shared pytest fixtures for MLPrep tests."""

from __future__ import annotations

import pandas as pd
import pytest

from mlprep.phases.phase1.config import Phase1Config
from mlprep.types import TaskType


@pytest.fixture
def clean_binary_classification_data() -> pd.DataFrame:
    """Return a clean binary-classification dataset."""

    return pd.DataFrame(
        {
            "customer_id": [
                "C001",
                "C002",
                "C003",
                "C004",
                "C005",
                "C006",
            ],
            "snapshot_date": [
                "2026-01-01",
                "2026-01-01",
                "2026-01-01",
                "2026-02-01",
                "2026-02-01",
                "2026-02-01",
            ],
            "age": [
                24,
                31,
                45,
                28,
                52,
                39,
            ],
            "monthly_income": [
                5000.0,
                7200.0,
                9100.0,
                6500.0,
                12000.0,
                8300.0,
            ],
            "city": [
                "Cairo",
                "Giza",
                "Alexandria",
                "Cairo",
                "Giza",
                "Alexandria",
            ],
            "churn": [
                0,
                1,
                0,
                1,
                0,
                1,
            ],
        }
    )


@pytest.fixture
def dirty_binary_classification_data() -> pd.DataFrame:
    """Return a dataset containing common Phase 1 issues."""

    return pd.DataFrame(
        {
            " Customer ID ": [
                "C001",
                "C002",
                "C003",
                "C003",
                "C005",
                "C006",
            ],
            "Snapshot-Date": [
                "01/01/2026",
                "01/01/2026",
                "01/01/2026",
                "01/01/2026",
                "01/02/2026",
                "invalid-date",
            ],
            "Age": [
                24,
                "31",
                "unknown",
                "unknown",
                52,
                None,
            ],
            "Monthly Income": [
                5000.0,
                7200.0,
                -999,
                -999,
                12000.0,
                8300.0,
            ],
            "City": [
                " Cairo ",
                "GIZA",
                "N/A",
                "N/A",
                "",
                None,
            ],
            "Email Address": [
                "first@example.com",
                "second@example.com",
                "third@example.com",
                "third@example.com",
                "fifth@example.com",
                "sixth@example.com",
            ],
            "Final Churn Status": [
                0,
                1,
                0,
                0,
                0,
                1,
            ],
            "Churn": [
                0,
                1,
                0,
                0,
                0,
                1,
            ],
        }
    )


@pytest.fixture
def regression_data() -> pd.DataFrame:
    """Return a clean regression dataset."""

    return pd.DataFrame(
        {
            "property_id": [
                "P001",
                "P002",
                "P003",
                "P004",
                "P005",
                "P006",
            ],
            "area": [
                80.0,
                120.0,
                95.0,
                150.0,
                110.0,
                200.0,
            ],
            "bedrooms": [
                2,
                3,
                2,
                4,
                3,
                5,
            ],
            "city": [
                "Cairo",
                "Cairo",
                "Giza",
                "Giza",
                "Alexandria",
                "Alexandria",
            ],
            "price": [
                1_500_000.0,
                2_100_000.0,
                1_700_000.0,
                2_900_000.0,
                2_000_000.0,
                3_800_000.0,
            ],
        }
    )


@pytest.fixture
def duplicate_row_data() -> pd.DataFrame:
    """Return a dataset containing exact and keyed duplicates."""

    return pd.DataFrame(
        {
            "customer_id": [
                "C001",
                "C002",
                "C002",
                "C003",
                "C003",
            ],
            "snapshot_date": [
                "2026-01-01",
                "2026-01-01",
                "2026-01-01",
                "2026-02-01",
                "2026-02-01",
            ],
            "age": [
                24,
                31,
                31,
                45,
                45,
            ],
            "churn": [
                0,
                1,
                1,
                0,
                1,
            ],
        }
    )


@pytest.fixture
def mixed_type_data() -> pd.DataFrame:
    """Return a dataset with mixed Python value types."""

    return pd.DataFrame(
        {
            "record_id": [
                "R001",
                "R002",
                "R003",
                "R004",
                "R005",
            ],
            "age": pd.Series(
                [
                    21,
                    "22",
                    "unknown",
                    35,
                    None,
                ],
                dtype="object",
            ),
            "active": pd.Series(
                [
                    True,
                    "yes",
                    1,
                    False,
                    "no",
                ],
                dtype="object",
            ),
            "target": [
                0,
                1,
                0,
                1,
                0,
            ],
        }
    )


@pytest.fixture
def temporal_data() -> pd.DataFrame:
    """Return a dataset containing temporal text values."""

    return pd.DataFrame(
        {
            "record_id": [
                "R001",
                "R002",
                "R003",
                "R004",
            ],
            "event_date": [
                "01/03/2026",
                "15/03/2026",
                "30/03/2026",
                "invalid-date",
            ],
            "outcome_date": [
                "10/03/2026",
                "20/03/2026",
                "25/03/2026",
                "01/04/2026",
            ],
            "target": [
                0,
                1,
                0,
                1,
            ],
        }
    )


@pytest.fixture
def sensitive_data() -> pd.DataFrame:
    """Return synthetic data containing sensitive-like patterns."""

    return pd.DataFrame(
        {
            "record_id": [
                "R001",
                "R002",
                "R003",
                "R004",
            ],
            "email": [
                "person1@example.com",
                "person2@example.com",
                "person3@example.com",
                "person4@example.com",
            ],
            "phone": [
                "+201001234567",
                "+201101234567",
                "+201201234567",
                "+201501234567",
            ],
            "ip_address": [
                "192.168.1.1",
                "10.0.0.1",
                "172.16.0.1",
                "8.8.8.8",
            ],
            "target": [
                0,
                1,
                0,
                1,
            ],
        }
    )


@pytest.fixture
def binary_phase1_config() -> Phase1Config:
    """Return a standard Phase 1 binary-classification config."""

    config = Phase1Config.create(
        task_type=TaskType.BINARY_CLASSIFICATION,
        target_column="churn",
        positive_class=1,
        unit_of_observation=(
            "One customer at one monthly snapshot"
        ),
        observation_key=[
            "customer_id",
            "snapshot_date",
        ],
        entity_columns=[
            "customer_id",
        ],
        group_columns=[
            "customer_id",
        ],
        prediction_timestamp_column=(
            "snapshot_date"
        ),
        identifier_columns=[
            "customer_id",
        ],
    )

    config.memory.level = "disabled"

    config.temporal_detection.convert_detected_columns = (
        True
    )

    return config


@pytest.fixture
def regression_phase1_config() -> Phase1Config:
    """Return a standard Phase 1 regression config."""

    config = Phase1Config.create(
        task_type=TaskType.REGRESSION,
        target_column="price",
        unit_of_observation="One property listing",
        observation_key=[
            "property_id",
        ],
        entity_columns=[
            "property_id",
        ],
        identifier_columns=[
            "property_id",
        ],
    )

    config.memory.level = "disabled"

    return config
from __future__ import annotations

import pytest

from mlprep.phases.phase4.indicators.indicator_naming import (
    IndicatorCollisionPolicy,
    IndicatorNameResolver,
    IndicatorNamingOptions,
)


def test_default_and_truncated_indicator_names() -> None:
    resolver = IndicatorNameResolver()
    limited = IndicatorNameResolver(
        IndicatorNamingOptions(
            suffix="_missing",
            maximum_name_length=10,
        )
    )

    assert resolver.base_name("amount") == "amount__was_missing"
    assert limited.base_name("amount") == "amount_mis"
    assert resolver.resolve(
        column="amount",
        existing_columns=[],
    ) == "amount__was_missing"


def test_indicator_name_validation_rejects_empty_column_and_invalid_options() -> None:
    with pytest.raises(ValueError, match="column cannot be empty"):
        IndicatorNameResolver().base_name(" ")

    with pytest.raises(ValueError, match="suffix cannot be empty"):
        IndicatorNamingOptions(suffix="")

    with pytest.raises(ValueError, match="maximum_collision_attempts"):
        IndicatorNamingOptions(maximum_collision_attempts=0)

    with pytest.raises(ValueError, match="maximum_name_length"):
        IndicatorNamingOptions(maximum_name_length=0)

    with pytest.raises(ValueError, match="must contain"):
        IndicatorNamingOptions(
            collision_policy=IndicatorCollisionPolicy.SUFFIX,
            collision_suffix_template="_collision",
        )


def test_collision_raise_and_reuse_policies() -> None:
    existing = {"amount__was_missing"}

    with pytest.raises(ValueError, match="name collision"):
        IndicatorNameResolver().resolve(
            column="amount",
            existing_columns=existing,
        )

    reused = IndicatorNameResolver(
        IndicatorNamingOptions(
            collision_policy=IndicatorCollisionPolicy.REUSE,
        )
    ).resolve(
        column="amount",
        existing_columns=existing,
        reserved_columns={"another_reserved_name"},
    )

    assert reused == "amount__was_missing"


def test_suffix_policy_skips_existing_and_reserved_names() -> None:
    resolver = IndicatorNameResolver(
        IndicatorNamingOptions(
            collision_policy=IndicatorCollisionPolicy.SUFFIX,
            maximum_collision_attempts=3,
        )
    )

    resolved = resolver.resolve(
        column="amount",
        existing_columns={"amount__was_missing"},
        reserved_columns={"amount__was_missing__1"},
    )

    assert resolved == "amount__was_missing__2"


def test_suffix_policy_reports_exhausted_collision_attempts() -> None:
    resolver = IndicatorNameResolver(
        IndicatorNamingOptions(
            collision_policy=IndicatorCollisionPolicy.SUFFIX,
            maximum_collision_attempts=1,
        )
    )

    with pytest.raises(RuntimeError, match="Unable to resolve"):
        resolver.resolve(
            column="amount",
            existing_columns={
                "amount__was_missing",
                "amount__was_missing__1",
            },
        )


def test_resolve_many_reserves_names_between_features() -> None:
    resolver = IndicatorNameResolver(
        IndicatorNamingOptions(
            suffix="_missing",
            collision_policy=IndicatorCollisionPolicy.SUFFIX,
        )
    )

    resolved = resolver.resolve_many(
        columns=["first", "second"],
        existing_columns={"first_missing"},
    )

    assert resolved == {
        "first": "first_missing__1",
        "second": "second_missing",
    }


def test_collision_suffix_obeys_maximum_name_length() -> None:
    resolver = IndicatorNameResolver(
        IndicatorNamingOptions(
            suffix="_missing",
            collision_policy=IndicatorCollisionPolicy.SUFFIX,
            maximum_name_length=12,
        )
    )

    resolved = resolver.resolve(
        column="feature",
        existing_columns={"feature_miss"},
    )

    assert resolved == "feature_m__1"

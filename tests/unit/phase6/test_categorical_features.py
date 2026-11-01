from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlprep.phases.phase6.categorical.categorical_pipeline import (
    CategoricalPipeline,
    CategoricalPipelineOptions,
)
from mlprep.phases.phase6.categorical.frequency_encoder import (
    FrequencyEncoder,
    FrequencyEncoderOptions,
    FrequencyEncoderState,
)
from mlprep.phases.phase6.categorical.hashing_encoder import (
    HashingEncoder,
    HashingEncoderOptions,
    HashingEncoderState,
)
from mlprep.phases.phase6.categorical.one_hot_encoder import (
    OneHotEncoder,
    OneHotEncoderOptions,
    OneHotEncoderState,
)
from mlprep.phases.phase6.categorical.ordinal_encoder import (
    OrdinalEncoder,
    OrdinalEncoderOptions,
    OrdinalEncoderState,
)
from mlprep.phases.phase6.categorical.rare_category_grouper import (
    RareCategoryGrouper,
    RareCategoryGrouperOptions,
    RareCategoryGrouperState,
)
from mlprep.phases.phase6.contract.transformation_plan import (
    CategoricalEncodingKind,
    FeatureFamily,
    TransformationPlan,
    TransformationPlanEntry,
)
from mlprep.phases.phase6.feature_engineering.boolean_features import (
    BooleanFeatureOptions,
    BooleanFeatureState,
    BooleanFeatureTransformer,
)
from mlprep.phases.phase6.feature_engineering.datetime_features import (
    DatetimeFeatureExtractor,
    DatetimeFeatureOptions,
    DatetimeFeatureState,
)
from mlprep.phases.phase6.feature_engineering.feature_lineage import (
    FeatureLineageState,
    FeatureLineageTracker,
)
from mlprep.phases.phase6.feature_engineering.text_features import (
    TextFeatureExtractor,
    TextFeatureOptions,
    TextFeatureState,
)


def test_one_hot_uses_training_vocabulary_and_stable_unknown_schema() -> None:
    training = pd.Series(
        ["b", "a", None, "b", "c"],
        index=[10, 11, 12, 13, 14],
        name="color",
        dtype="object",
    )
    original = training.copy(deep=True)
    encoder = OneHotEncoder(
        column="color",
        options=OneHotEncoderOptions(
            maximum_categories=3,
            handle_unknown="indicator",
        ),
    )
    encoded = encoder.fit_transform(training)
    restored = OneHotEncoder(
        column="color",
        options=OneHotEncoderOptions(
            maximum_categories=3,
            handle_unknown="indicator",
        ),
    )
    restored.state = OneHotEncoderState.model_validate_json(
        encoder.get_state().model_dump_json()
    )
    validation = encoder.transform(
        pd.Series(["c", None], index=[20, 21], name="color")
    )
    restored_validation = restored.transform(
        pd.Series(["c", None], index=[20, 21], name="color")
    )

    assert encoder.state.categories == ["b", "__MISSING__", "a", "__UNKNOWN__"]
    assert encoded.columns.tolist() == [
        "color__onehot__b",
        "color__onehot____MISSING__",
        "color__onehot__a",
        "color__onehot____UNKNOWN__",
    ]
    assert validation.columns.tolist() == encoded.columns.tolist()
    pd.testing.assert_frame_equal(
        restored_validation,
        validation,
    )
    assert validation.iloc[0].tolist() == [0.0, 0.0, 0.0, 1.0]
    assert validation.iloc[1].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert training.equals(original)
    assert encoder.unknown_mask(
        pd.Series(["c"], name="color")
    ).tolist() == [True]
    assert encoder.state.fitted_on_training_only


def test_one_hot_rejects_sanitized_output_name_collisions() -> None:
    encoder = OneHotEncoder(
        column="category",
        options=OneHotEncoderOptions(
            handle_unknown="ignore",
        ),
    )

    with pytest.raises(ValueError, match="collision"):
        encoder.fit(
            pd.Series(["a b", "a/b"], name="category")
        )


def test_one_hot_limits_high_cardinality_vocabulary_deterministically() -> None:
    encoder = OneHotEncoder(
        column="category",
        options=OneHotEncoderOptions(
            maximum_categories=2,
            handle_unknown="indicator",
        ),
    ).fit(
        pd.Series(
            ["z", "a", "b", "a"],
            name="category",
        )
    )

    encoded = encoder.transform(
        pd.Series(
            ["a", "b", "z", "new"],
            name="category",
        )
    )

    assert encoder.state.categories == ["a", "b", "__UNKNOWN__"]
    assert encoded.to_numpy().tolist() == [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ]


def test_ordinal_and_frequency_encoders_use_fixed_training_mappings() -> None:
    training = pd.Series(
        ["low", "high", "low", None],
        index=[3, 5, 8, 13],
        name="level",
    )
    original = training.copy(deep=True)
    ordinal = OrdinalEncoder(
        column="level",
        category_order=["low", "medium", "high"],
        options=OrdinalEncoderOptions(
            unknown_value=-9,
            missing_value=-8,
        ),
    ).fit(training)
    ordinal_result = ordinal.transform(
        pd.Series(["high", "new", None], index=[20, 21, 22], name="level")
    )
    frequency = FrequencyEncoder(
        column="level",
        options=FrequencyEncoderOptions(
            mode="count",
            unknown_value=-1,
        ),
    ).fit(training)
    frequency_result = frequency.transform(
        pd.Series(["low", "new", None], index=[20, 21, 22], name="level")
    )

    assert ordinal_result.iloc[:, 0].tolist() == [2, -9, -8]
    assert frequency_result.iloc[:, 0].tolist() == [2.0, -1.0, 1.0]
    assert ordinal.unknown_mask(
        pd.Series(["new"], name="level")
    ).tolist() == [True]
    assert ordinal.state.fitted_on_training_only
    assert frequency.state.fitted_on_training_only
    assert training.equals(original)


def test_ordinal_unknown_error_policy_rejects_unseen_values() -> None:
    encoder = OrdinalEncoder(
        column="level",
        category_order=["low", "high"],
        options=OrdinalEncoderOptions(
            handle_unknown="error",
        ),
    ).fit(
        pd.Series(["low", "high"], name="level")
    )

    with pytest.raises(ValueError, match="Unknown ordinal categories"):
        encoder.transform(
            pd.Series(["medium"], name="level")
        )


def test_ordinal_state_round_trip_preserves_explicit_order() -> None:
    options = OrdinalEncoderOptions(
        unknown_value=-9,
        missing_value=-8,
    )
    encoder = OrdinalEncoder(
        column="level",
        category_order=["low", "medium", "high"],
        options=options,
    ).fit(
        pd.Series(["low", "high"], name="level")
    )
    restored = OrdinalEncoder(
        column="level",
        category_order=["low", "medium", "high"],
        options=options,
    )
    restored.state = OrdinalEncoderState.model_validate_json(
        encoder.get_state().model_dump_json()
    )

    actual = restored.transform(
        pd.Series(["high", "unknown", None], name="level")
    )

    assert actual["level__ordinal"].tolist() == [2, -9, -8]
    assert restored.state.fitted_on_training_only


def test_frequency_encoding_uses_training_frequencies_and_restores_state() -> None:
    training = pd.Series(
        ["a", "a", "b"],
        index=[4, 5, 6],
        name="category",
    )
    encoder = FrequencyEncoder(
        column="category",
    ).fit(training)
    state = FrequencyEncoderState.model_validate_json(
        encoder.get_state().model_dump_json()
    )
    restored = FrequencyEncoder(
        column="category",
    )
    restored.state = state

    actual = restored.transform(
        pd.Series(
            ["a", "b", "unseen"],
            index=[10, 11, 12],
            name="category",
        )
    )

    assert actual["category__frequency"].tolist() == [
        pytest.approx(2 / 3),
        pytest.approx(1 / 3),
        0.0,
    ]
    assert state.mapping == {"a": pytest.approx(2 / 3), "b": pytest.approx(1 / 3)}
    assert state.training_row_count == 3


def test_hashing_is_deterministic_and_keeps_fixed_schema() -> None:
    training = pd.Series(
        ["alpha", "beta", "alpha"],
        index=[4, 8, 15],
        name="token",
    )
    original = training.copy(deep=True)
    options = HashingEncoderOptions(
        bucket_count=8,
        seed="stable-seed",
        signed_hashing=True,
    )
    first = HashingEncoder(
        column="token",
        options=options,
    ).fit(training)
    second = HashingEncoder(
        column="token",
        options=HashingEncoderOptions(
            bucket_count=8,
            seed="stable-seed",
            signed_hashing=True,
        ),
    ).fit(training)

    first_result = first.transform(training)
    second_result = second.transform(training)
    unseen = first.transform(
        pd.Series(["new"], index=[25], name="token")
    )

    pd.testing.assert_frame_equal(first_result, second_result)
    assert unseen.columns.tolist() == first_result.columns.tolist()
    assert unseen.index.tolist() == [25]
    assert first.state.fitted_on_training_only
    assert training.equals(original)


def test_hashing_state_round_trip_preserves_duplicate_index_behavior() -> None:
    training = pd.Series(
        ["alpha", "beta"],
        index=[1, 1],
        name="token",
    )
    options = HashingEncoderOptions(bucket_count=8)
    encoder = HashingEncoder(
        column="token",
        options=options,
    ).fit(training)
    restored = HashingEncoder(
        column="token",
        options=options,
    )
    restored.state = HashingEncoderState.model_validate_json(
        encoder.get_state().model_dump_json()
    )

    expected = encoder.transform(training)
    actual = restored.transform(training)

    pd.testing.assert_frame_equal(actual, expected)
    assert actual.sum(axis=1).tolist() == [1.0, 1.0]


def test_hashing_preserves_rows_with_duplicate_index_labels() -> None:
    training = pd.Series(
        ["alpha", "beta"],
        index=[1, 1],
        name="token",
    )
    encoder = HashingEncoder(
        column="token",
        options=HashingEncoderOptions(
            bucket_count=8,
        ),
    ).fit(training)

    encoded = encoder.transform(training)

    assert encoded.index.tolist() == [1, 1]
    assert encoded.sum(axis=1).tolist() == [1.0, 1.0]


def test_rare_category_grouper_preserves_missing_and_groups_unseen_values() -> None:
    training = pd.Series(
        ["common", "common", "rare", None],
        index=[1, 2, 3, 4],
        name="segment",
    )
    original = training.copy(deep=True)
    grouper = RareCategoryGrouper(
        column="segment",
        options=RareCategoryGrouperOptions(
            minimum_count=2,
        ),
    ).fit(training)
    transformed = grouper.transform(
        pd.Series(
            ["common", "rare", "unseen", None],
            index=[10, 11, 12, 13],
            name="segment",
        )
    )

    assert transformed.tolist() == [
        "common",
        "__RARE__",
        "__RARE__",
        "__MISSING__",
    ]
    assert grouper.state.rare_categories == ["rare"]
    assert grouper.state.fitted_on_training_only
    assert transformed.index.tolist() == [10, 11, 12, 13]
    assert training.equals(original)


def test_rare_category_state_round_trip_preserves_training_vocabulary() -> None:
    training = pd.Series(
        ["common", "common", "rare"],
        name="segment",
    )
    options = RareCategoryGrouperOptions(minimum_count=2)
    grouper = RareCategoryGrouper(
        column="segment",
        options=options,
    ).fit(training)
    restored = RareCategoryGrouper(
        column="segment",
        options=options,
    )
    restored.state = RareCategoryGrouperState.model_validate_json(
        grouper.get_state().model_dump_json()
    )

    actual = restored.transform(
        pd.Series(["common", "rare", "new"], name="segment")
    )

    assert actual.tolist() == ["common", "__RARE__", "__RARE__"]
    assert restored.state.fitted_on_training_only


def test_categorical_pipeline_keeps_train_schema_and_reports_unknowns() -> None:
    training = pd.DataFrame(
        {
            "color": ["blue", "red", "blue"],
            "shape": ["round", "square", "round"],
        },
        index=[7, 8, 9],
    )
    original = training.copy(deep=True)
    plan = TransformationPlan(
        entries={
            "color": TransformationPlanEntry(
                column="color",
                feature_family=FeatureFamily.CATEGORICAL,
                categorical_encoding=CategoricalEncodingKind.ONE_HOT,
                reason="one-hot color",
            ),
            "shape": TransformationPlanEntry(
                column="shape",
                feature_family=FeatureFamily.CATEGORICAL,
                categorical_encoding=CategoricalEncodingKind.FREQUENCY,
                reason="frequency shape",
            ),
        },
        categorical_columns=["color", "shape"],
    )
    pipeline = CategoricalPipeline(
        CategoricalPipelineOptions()
    ).fit(frame=training, plan=plan)
    fitted_state = {
        column: state.model_dump(mode="json")
        for column, state in pipeline.states.items()
    }
    validation = pd.DataFrame(
        {
            "color": ["green"],
            "shape": ["triangle"],
        },
        index=[20],
    )
    result = pipeline.transform(frame=validation)

    assert result.frame.index.tolist() == [20]
    assert result.output_columns == pipeline.output_columns()
    assert result.unknown_categories["color"].unknown_count == 1
    assert result.unknown_categories["shape"].unknown_count == 1
    assert result.states["color"].fitted_on_training_only
    assert {
        column: state.model_dump(mode="json")
        for column, state in pipeline.states.items()
    } == fitted_state
    assert result.frame.columns.tolist() == [
        "color__onehot__blue",
        "color__onehot__red",
        "color__onehot____UNKNOWN__",
        "shape__frequency",
    ]
    assert training.equals(original)
    assert np.isfinite(result.frame.to_numpy(dtype=float)).all()


def test_categorical_pipeline_rejects_cross_feature_name_collisions() -> None:
    training = pd.DataFrame(
        {
            "first": ["same", "other"],
            "second": ["same", "other"],
        }
    )
    plan = TransformationPlan(
        entries={
            column: TransformationPlanEntry(
                column=column,
                feature_family=FeatureFamily.CATEGORICAL,
                categorical_encoding=CategoricalEncodingKind.ONE_HOT,
                output_prefix="shared",
                reason="one-hot feature",
            )
            for column in ["first", "second"]
        },
        categorical_columns=["first", "second"],
    )

    with pytest.raises(ValueError, match="output-column collision"):
        CategoricalPipeline().fit(
            frame=training,
            plan=plan,
        )


def test_boolean_normalization_handles_vocabularies_and_missing_by_policy() -> None:
    training = pd.Series(
        [" YES ", "no", "true", "0"],
        index=[2, 4, 6, 8],
        name="active",
    )
    original = training.copy(deep=True)
    transformer = BooleanFeatureTransformer(
        column="active",
        options=BooleanFeatureOptions(),
    ).fit(training)
    transformed = transformer.transform(
        pd.Series(["Y", "FALSE"], index=[10, 11], name="active")
    )

    assert transformed.iloc[:, 0].tolist() == [1, 0]
    assert transformed.index.tolist() == [10, 11]
    assert transformer.state.fitted_on_training_only
    assert training.equals(original)

    missing = BooleanFeatureTransformer(
        column="active",
        options=BooleanFeatureOptions(
            missing_policy="use_value",
            missing_value=-7,
        ),
    ).fit(pd.Series(["yes", None], name="active"))
    assert missing.transform(
        pd.Series([None], name="active")
    ).iloc[0, 0] == -7


def test_boolean_unknown_values_follow_configured_policy() -> None:
    transformer = BooleanFeatureTransformer(
        column="active",
        options=BooleanFeatureOptions(
            unknown_policy="use_value",
            unknown_value=9,
            missing_policy="use_value",
            missing_value=-7,
        ),
    ).fit(
        pd.Series(["yes", "maybe", None], name="active")
    )

    result = transformer.transform(
        pd.Series(["yes", "maybe", None], name="active")
    )

    assert result["active__boolean"].tolist() == [1, 9, -7]
    assert transformer.state.training_unknown_count == 1


def test_boolean_state_round_trip_preserves_normalization() -> None:
    options = BooleanFeatureOptions()
    transformer = BooleanFeatureTransformer(
        column="active",
        options=options,
    ).fit(
        pd.Series(["yes", "no"], name="active")
    )
    restored = BooleanFeatureTransformer(
        column="active",
        options=options,
    )
    restored.state = BooleanFeatureState.model_validate_json(
        transformer.get_state().model_dump_json()
    )

    actual = restored.transform(
        pd.Series([" TRUE ", "N"], name="active")
    )

    assert actual["active__boolean"].tolist() == [1, 0]
    assert restored.state.fitted_on_training_only


def test_text_features_are_structural_finite_and_restore_from_state() -> None:
    training = pd.Series(
        ["Hello, 42", None],
        index=[30, 31],
        name="message",
    )
    original = training.copy(deep=True)
    options = TextFeatureOptions(
        features=[
            "character_count",
            "word_count",
            "digit_count",
            "punctuation_count",
            "unique_word_count",
            "average_word_length",
        ],
        missing_replacement="",
    )
    extractor = TextFeatureExtractor(
        column="message",
        options=options,
    ).fit(training)
    expected = extractor.transform(training)
    restored = TextFeatureExtractor(
        column="message",
        options=options,
    )
    restored.state = TextFeatureState.model_validate_json(
        extractor.get_state().model_dump_json()
    )
    actual = restored.transform(training)

    pd.testing.assert_frame_equal(actual, expected)
    assert actual.loc[30, "message__text__word_count"] == 2
    assert actual.loc[30, "message__text__digit_count"] == 2
    assert actual.loc[30, "message__text__unique_word_count"] == 2
    assert np.isfinite(actual.to_numpy(dtype=float)).all()
    assert extractor.state.fitted_on_training_only
    assert training.equals(original)


def test_datetime_cycles_elapsed_state_and_input_immutability() -> None:
    training = pd.Series(
        pd.to_datetime(
            [
                "2024-01-01 00:00:00",
                "2024-01-01 12:00:00",
            ]
        ),
        index=[40, 41],
        name="event_time",
    )
    original = training.copy(deep=True)
    options = DatetimeFeatureOptions(
        features=[
            "hour_sin",
            "hour_cos",
            "day_of_week_sin",
            "day_of_week_cos",
            "month_sin",
            "month_cos",
            "elapsed_days",
        ],
    )
    extractor = DatetimeFeatureExtractor(
        column="event_time",
        options=options,
    ).fit(training)
    expected = extractor.transform(training)
    restored = DatetimeFeatureExtractor(
        column="event_time",
        options=options,
    )
    restored.state = DatetimeFeatureState.model_validate_json(
        extractor.get_state().model_dump_json()
    )
    actual = restored.transform(training)

    pd.testing.assert_frame_equal(actual, expected)
    assert actual.loc[40, "event_time__datetime__hour_sin"] == pytest.approx(0.0)
    assert actual.loc[40, "event_time__datetime__hour_cos"] == pytest.approx(1.0)
    assert actual.loc[41, "event_time__datetime__hour_sin"] == pytest.approx(0.0)
    assert actual.loc[41, "event_time__datetime__elapsed_days"] == pytest.approx(0.5)
    assert actual.dtypes.iloc[:6].eq("float64").all()
    assert extractor.state.fitted_on_training_only
    assert training.equals(original)


def test_datetime_cyclical_features_remain_floating_for_integer_output_option() -> None:
    extractor = DatetimeFeatureExtractor(
        column="timestamp",
        options=DatetimeFeatureOptions(
            features=["month_sin", "month_cos"],
            output_dtype="int64",
        ),
    )
    result = extractor.fit_transform(
        pd.Series(
            pd.to_datetime(["2024-01-01", "2024-04-01"]),
            name="timestamp",
        )
    )

    assert result.iloc[0, 0] == pytest.approx(0.0)
    assert result.iloc[0, 1] == pytest.approx(1.0)
    assert result.iloc[1, 0] == pytest.approx(1.0)
    assert result.iloc[1, 1] == pytest.approx(0.0)
    assert result.dtypes.tolist() == ["float64", "float64"]


def test_feature_lineage_maps_all_sources_and_round_trips() -> None:
    tracker = FeatureLineageTracker(
        source_columns=["city", "country"]
    )
    tracker.register(
        output_column="city__country",
        source_columns=["city", "country"],
        transformation_steps=["categorical_join", "one_hot"],
        feature_family="categorical",
        metadata={"training_only": True},
    )
    tracker.register_passthrough(
        column="age",
        feature_family="numeric",
    )
    restored = FeatureLineageState.model_validate(
        tracker.to_dictionary()
    )

    assert tracker.outputs_for("city") == ["city__country"]
    assert tracker.sources_for("city__country") == ["city", "country"]
    assert restored.output_columns == [
        "city__country",
        "age",
    ]
    assert restored.entries["city__country"].transformation_steps == [
        "categorical_join",
        "one_hot",
    ]
    tracker.validate_output_schema(["city__country", "age"])
    with pytest.raises(ValueError, match="differs"):
        tracker.validate_output_schema(["city__country"])

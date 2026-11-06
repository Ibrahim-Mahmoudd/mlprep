from __future__ import annotations

import json

import pytest

from mlprep.phases.phase8.config_loader import Phase8ConfigLoader
from mlprep.phases.phase8.contract import (
    BalancingStrategy,
    ClassWeightMethod,
    SamplingStrategyKind,
)


def test_phase8_loader_builds_nested_options_and_enum_values() -> None:
    config = Phase8ConfigLoader.from_dict(
        {
            "execution": {"enabled": False},
            "plan": {
                "strategy": "random_oversampling",
                "sampling_strategy": "custom",
                "class_weight_method": "custom",
            },
            "strategy_recommender": {
                "severe_imbalance_strategy": "random_oversampling",
            },
            "class_weights": {
                "method": "custom",
            },
            "distribution_analysis": {
                "balanced_count_tolerance": 2,
            },
            "validation": {
                "distribution": {"minimum_output_class_count": 2},
            },
            "pipeline": {
                "require_nonempty_training_output": False,
                "metrics": {},
                "recommender": {},
                "class_weights": {},
                "sample_weights": {},
                "partition_guard": {},
            },
        }
    )

    assert config.execution.enabled is False
    assert config.plan.strategy is BalancingStrategy.RANDOM_OVERSAMPLING
    assert config.plan.sampling_strategy is SamplingStrategyKind.CUSTOM
    assert config.plan.class_weight_method is ClassWeightMethod.CUSTOM
    assert (
        config.strategy_recommender.severe_imbalance_strategy
        is BalancingStrategy.RANDOM_OVERSAMPLING
    )
    assert config.class_weights.method is ClassWeightMethod.CUSTOM
    assert config.validation.distribution.minimum_output_class_count == 2
    assert config.pipeline.require_nonempty_training_output is False
    assert config.distribution_analysis.balanced_count_tolerance == 2


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"unknown": {}}, "Unknown Phase 8 configuration sections"),
        ({"plan": {"unknown_option": True}}, "unexpected keyword"),
        (
            {"validation": {"unknown": {}}},
            "Unknown Phase 8 validation options",
        ),
        ({"plan": []}, "Configuration section 'plan'"),
    ],
)
def test_phase8_loader_rejects_unknown_or_malformed_options(
    values: dict[str, object],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        Phase8ConfigLoader.from_dict(values)


def test_phase8_loader_loads_yaml_with_recursive_and_dotted_overrides(
    tmp_path,
) -> None:
    path = tmp_path / "phase8.yaml"
    path.write_text(
        "execution:\n  enabled: true\nplan:\n  random_seed: 5\n",
        encoding="utf-8",
    )

    config = Phase8ConfigLoader.load(
        path,
        overrides={"execution": {"enabled": False}},
        dotted_overrides={
            "plan.random_seed": 13,
            "reporting.include_phase_state": True,
        },
    )

    assert config.execution.enabled is False
    assert config.plan.random_seed == 13
    assert config.reporting.include_phase_state is True


def test_phase8_loader_json_round_trip_and_save_guards(tmp_path) -> None:
    config = Phase8ConfigLoader.from_dict(
        {
            "execution": {"enabled": False},
            "plan": {
                "strategy": "random_undersampling",
                "random_seed": 41,
            },
            "reporting": {"include_phase_state": True},
        }
    )
    path = Phase8ConfigLoader.save(config, tmp_path / "nested" / "config.json")

    assert json.loads(path.read_text(encoding="utf-8"))["execution"]["enabled"] is False
    assert Phase8ConfigLoader.load(path) == config
    with pytest.raises(FileExistsError):
        Phase8ConfigLoader.save(config, path)

    replaced = Phase8ConfigLoader.save(config, path, overwrite=True)
    assert Phase8ConfigLoader.load(replaced) == config

    yaml_path = Phase8ConfigLoader.save(
        config,
        tmp_path / "config.yaml",
        config_format="yaml",
    )
    assert Phase8ConfigLoader.load(yaml_path) == config
    with pytest.raises(ValueError, match="Cannot determine"):
        Phase8ConfigLoader.save(config, tmp_path / "config.txt")
    explicitly_formatted = Phase8ConfigLoader.save(
        config,
        tmp_path / "config.txt",
        config_format="json",
    )
    assert json.loads(
        explicitly_formatted.read_text(encoding="utf-8")
    ) == config.to_dict()


def test_phase8_loader_validates_file_errors_and_safe_yaml(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        Phase8ConfigLoader.load_raw(tmp_path / "missing.yaml")
    with pytest.raises(IsADirectoryError):
        Phase8ConfigLoader.load_raw(tmp_path)

    unsupported = tmp_path / "config.toml"
    unsupported.write_text("enabled = true", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported configuration extension"):
        Phase8ConfigLoader.load_raw(unsupported)

    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        Phase8ConfigLoader.load_raw(empty)

    scalar = tmp_path / "scalar.yaml"
    scalar.write_text("- item", encoding="utf-8")
    with pytest.raises(TypeError, match="root must be a mapping"):
        Phase8ConfigLoader.load_raw(scalar)

    invalid_yaml = tmp_path / "invalid.yaml"
    invalid_yaml.write_text("plan: [", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid YAML"):
        Phase8ConfigLoader.load_raw(invalid_yaml)

    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON"):
        Phase8ConfigLoader.load_raw(invalid_json)


def test_phase8_loader_merge_and_dotted_override_validation() -> None:
    base = {"plan": {"seed": 1, "strategy": "none"}}
    merged = Phase8ConfigLoader.merge(
        base=base,
        override={"plan": {"seed": 2}},
    )
    assert merged == {"plan": {"seed": 2, "strategy": "none"}}
    assert base["plan"]["seed"] == 1

    dotted = Phase8ConfigLoader.apply_dot_overrides(
        values={},
        overrides={"plan.random_seed": 9},
    )
    assert dotted == {"plan": {"random_seed": 9}}
    with pytest.raises(ValueError, match="Invalid dotted path"):
        Phase8ConfigLoader.apply_dot_overrides(
            values={},
            overrides={"plan..seed": 1},
        )
    with pytest.raises(TypeError, match="is not a mapping"):
        Phase8ConfigLoader.apply_dot_overrides(
            values={"plan": 1},
            overrides={"plan.seed": 1},
        )
    with pytest.raises(TypeError, match="must be a mapping"):
        Phase8ConfigLoader.from_dict({"plan": []})

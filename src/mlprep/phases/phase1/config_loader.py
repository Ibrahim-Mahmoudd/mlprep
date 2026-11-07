"""Configuration loading and validation utilities for Phase 1."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, ClassVar, Literal

import yaml

from mlprep.phases.phase1.config import Phase1Config

ConfigFormat = Literal[
    "yaml",
    "json",
]


class Phase1ConfigLoader:
    """Load, merge, validate, and save Phase 1 configurations."""

    supported_yaml_extensions: ClassVar[frozenset[str]] = frozenset({
        ".yaml",
        ".yml",
    })

    supported_json_extensions: ClassVar[frozenset[str]] = frozenset({
        ".json",
    })

    supported_extensions: ClassVar[frozenset[str]] = (
        supported_yaml_extensions
        | supported_json_extensions
    )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        overrides: Mapping[str, Any] | None = None,
    ) -> Phase1Config:
        """Load a Phase 1 configuration from YAML or JSON."""

        input_path = cls._validate_input_path(path)

        raw_config = cls.load_raw(input_path)

        if overrides:
            raw_config = cls.merge(
                base=raw_config,
                override=dict(overrides),
            )

        return Phase1Config.from_dict(raw_config)

    @classmethod
    def load_raw(
        cls,
        path: str | Path,
    ) -> dict[str, Any]:
        """Load raw configuration values without creating models."""

        input_path = cls._validate_input_path(path)

        extension = input_path.suffix.casefold()

        if extension in cls.supported_yaml_extensions:
            return cls._load_yaml(input_path)

        if extension in cls.supported_json_extensions:
            return cls._load_json(input_path)

        supported = ", ".join(
            sorted(cls.supported_extensions)
        )

        raise ValueError(
            f"Unsupported configuration extension "
            f"{extension!r}. Supported extensions: {supported}"
        )

    @classmethod
    def save(
        cls,
        config: Phase1Config,
        path: str | Path,
        *,
        config_format: ConfigFormat | None = None,
        overwrite: bool = False,
    ) -> Path:
        """Save a resolved Phase 1 configuration."""

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
            or cls._format_from_extension(output_path)
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        config_values = config.to_dict()

        if resolved_format == "yaml":
            serialized_content = yaml.safe_dump(
                config_values,
                sort_keys=False,
                allow_unicode=True,
            )
        else:
            serialized_content = json.dumps(
                config_values,
                indent=2,
                ensure_ascii=False,
                default=str,
            )

        output_path.write_text(
            serialized_content,
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
        """Recursively merge configuration mappings."""

        merged = deepcopy(dict(base))

        for key, override_value in override.items():
            base_value = merged.get(key)

            if (
                isinstance(base_value, Mapping)
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
        values: Mapping[str, Any],
        overrides: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Apply overrides using dotted configuration paths.

        Example:
            {
                "column_names.case_style": "lower",
                "memory.level": "disabled",
            }
        """

        updated_values = deepcopy(
            dict(values)
        )

        for dotted_path, override_value in overrides.items():
            cls._set_dotted_value(
                values=updated_values,
                dotted_path=dotted_path,
                value=override_value,
            )

        return updated_values

    @classmethod
    def load_with_dot_overrides(
        cls,
        path: str | Path,
        overrides: Mapping[str, Any],
    ) -> Phase1Config:
        """Load configuration and apply dotted-path overrides."""

        raw_config = cls.load_raw(path)

        updated_config = cls.apply_dot_overrides(
            values=raw_config,
            overrides=overrides,
        )

        return Phase1Config.from_dict(
            updated_config
        )

    @classmethod
    def validate_file(
        cls,
        path: str | Path,
    ) -> list[object]:
        """Validate a configuration file and return error messages."""

        errors: list[str] = []

        try:
            cls.load(path)
        except (
            FileNotFoundError,
            IsADirectoryError,
            PermissionError,
            TypeError,
            ValueError,
            yaml.YAMLError,
            json.JSONDecodeError,
        ) as error:
            errors.append(str(error))

        return errors

    @classmethod
    def create_template(
        cls,
        config: Phase1Config,
        path: str | Path,
        *,
        overwrite: bool = False,
    ) -> Path:
        """Create a YAML configuration template."""

        output_path = Path(path)

        if output_path.suffix.casefold() not in {
            ".yaml",
            ".yml",
        }:
            output_path = output_path.with_suffix(
                ".yaml"
            )

        return cls.save(
            config=config,
            path=output_path,
            config_format="yaml",
            overwrite=overwrite,
        )

    @staticmethod
    def _load_yaml(
        path: Path,
    ) -> dict[str, Any]:
        """Load and validate a YAML configuration mapping."""

        try:
            loaded_content = yaml.safe_load(
                path.read_text(
                    encoding="utf-8"
                )
            )
        except yaml.YAMLError as error:
            raise ValueError(
                f"Invalid YAML configuration: {error}"
            ) from error

        return Phase1ConfigLoader._validate_root_mapping(
            loaded_content=loaded_content,
            path=path,
        )

    @staticmethod
    def _load_json(
        path: Path,
    ) -> dict[str, Any]:
        """Load and validate a JSON configuration mapping."""

        try:
            loaded_content = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON configuration: {error}"
            ) from error

        return Phase1ConfigLoader._validate_root_mapping(
            loaded_content=loaded_content,
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
                f"{extension!r}. Supported extensions: "
                f"{supported}"
            )

        return input_path

    @staticmethod
    def _validate_root_mapping(
        *,
        loaded_content: object,
        path: Path,
    ) -> dict[str, Any]:
        """Validate the root configuration object."""

        if loaded_content is None:
            raise ValueError(
                f"Configuration file is empty: {path}"
            )

        if not isinstance(
            loaded_content,
            dict,
        ):
            raise TypeError(
                "The configuration root must be a mapping, "
                f"but received {type(loaded_content).__name__!r}."
            )

        normalized_content: dict[str, Any] = {}

        for key, value in loaded_content.items():
            if not isinstance(key, str):
                raise TypeError(
                    "Every root configuration key must be "
                    f"a string, but received {type(key).__name__!r}."
                )

            normalized_content[key] = value

        return normalized_content

    @staticmethod
    def _format_from_extension(
        path: Path,
    ) -> ConfigFormat:
        """Resolve a configuration format from a file extension."""

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
            f"extension {extension!r}."
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
                f"Invalid dotted override path: "
                f"{dotted_path!r}"
            )

        current_level = values

        for path_part in path_parts[:-1]:
            existing_value = current_level.get(
                path_part
            )

            if existing_value is None:
                new_mapping: dict[str, Any] = {}

                current_level[path_part] = (
                    new_mapping
                )

                current_level = new_mapping
                continue

            if not isinstance(
                existing_value,
                dict,
            ):
                raise TypeError(
                    f"Cannot set {dotted_path!r} because "
                    f"{path_part!r} is not a mapping."
                )

            current_level = existing_value

        current_level[path_parts[-1]] = deepcopy(
            value
        )

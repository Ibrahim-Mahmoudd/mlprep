# Unified Configuration

MLPrep accepts YAML and JSON configuration files with a flat root containing
`phase1` through `phase9`, `cross_validation`, and `pipeline`.

The canonical version field is `config_version`. Existing files using
`schema_version` remain readable for compatibility; when loaded, resolved
configuration is serialized with `config_version`. If both keys are present,
their values must match. The current supported version is `"1"`.

Overrides are applied in this order:

1. Values in the configuration file.
2. Dotted overrides passed to `MLPrepConfigLoader.load`.
3. Environment overrides using the `MLPREP_` prefix and double underscores
   for nested keys, such as `MLPREP_PHASE2__RANDOM_SEED`.

Unknown root and nested keys are rejected instead of silently ignored.

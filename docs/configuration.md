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

## Data sources

An optional `data_source` section records the typed source configuration and
participates in resolved-configuration fingerprints:

```yaml
data_source:
  type: csv
  config:
    path: data/customers.csv
    encoding: utf-8
    expected_schema:
      customer_id: integer
```

Supported types are `dataframe`, `csv`, `json`, `jsonl`, `parquet`, `sql`, and
`api`. Credentials are referenced by environment-variable name only; token and
credential values are never serialized into resolved configuration or state.
Parquet and SQL adapters use optional runtime dependencies. SQL chunks are
combined into a DataFrame before pipeline execution, so this interface does not
provide end-to-end streaming or out-of-core processing.

Phase 6 numeric transformation overrides accept `log1p`, `signed_log1p`,
`box_cox`, and `yeo_johnson`. Box-Cox requires strictly positive values in
both training and transformed partitions; it does not shift zero or negative
values automatically. Its fitting options are configured under
`phase6.numeric.box_cox` (lambda bounds, search steps, constant tolerance, and
output dtype preservation).

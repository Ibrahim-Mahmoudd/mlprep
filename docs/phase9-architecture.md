# Phase 9: Model Handoff and Governance

## Responsibility

Phase 9 consumes a ready Phase 8 result and produces an auditable, model-ready
handoff. It is the boundary between data preparation and downstream model
training. It does not fit, evaluate, or select a model.

## Inputs and safety invariants

- Accept only a Phase 8 result whose readiness decision is ready and whose
  train, validation, and test partitions satisfy the Phase 8 governance checks.
- Preserve the three partitions and their row order, targets, feature schemas,
  row identifiers, and weights. Validation and test data must remain unchanged.
- Carry forward training-only fitted state and feature/row lineage as references
  and metadata; never refit transformations or access test data to fit state.
- Reject incomplete or contradictory state instead of manufacturing a
  success-shaped handoff.

## Outputs

The handoff contains the governed partitions, target and feature declarations,
training weights when present, lineage/fingerprint references, a readiness
decision, findings, and a JSON-compatible summary. Dataset payloads remain
available as in-memory partitions; the summary is metadata, not a substitute
for serializing data or fitted estimators.

## Non-goals

Phase 9 does not train models, tune hyperparameters, compute model metrics, or
write files implicitly. Persistence and model-specific behavior remain the
responsibility of downstream consumers.

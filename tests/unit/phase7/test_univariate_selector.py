from __future__ import annotations

import pandas as pd
import pytest

from mlprep.phases.phase7.contract.selection_plan import (
    UnivariateMethod,
)
from mlprep.phases.phase7.selectors.base import (
    SelectorFitContext,
)
from mlprep.phases.phase7.selectors.univariate_selector import (
    UnivariateSelector,
)


def _classification_data() -> tuple[pd.DataFrame, pd.Series]:
    target = pd.Series([0, 0, 0, 0, 1, 1, 1, 1], name="target")
    features = pd.DataFrame(
        {
            "signal": [0.0, 0.1, 0.2, 0.3, 8.0, 8.1, 8.2, 8.3],
            "noise": [4.0, 1.0, 7.0, 3.0, 2.0, 6.0, 0.0, 5.0],
        }
    )
    return features, target


def test_univariate_selector_ranks_signal_and_honors_top_k() -> None:
    features, target = _classification_data()
    selector = UnivariateSelector(
        task="classification",
        method=UnivariateMethod.ANOVA,
        top_k=1,
    )

    selector.fit(
        features=features,
        target=target,
        context=SelectorFitContext.TRAINING_SUPERVISED,
        target_access_authorized=True,
    )

    assert selector.selected_features() == ["signal"]
    assert selector.rejected_features() == ["noise"]
    assert selector.rank_features()[0] == "signal"
    assert selector.state.fitted_on_training_only


def test_univariate_selector_requires_training_context_and_authorization() -> None:
    features, target = _classification_data()
    selector = UnivariateSelector(
        task="classification",
        method=UnivariateMethod.ANOVA,
    )

    with pytest.raises(PermissionError, match="training-supervised"):
        selector.fit(
            features=features,
            target=target,
            context=SelectorFitContext.VALIDATION,
            target_access_authorized=True,
        )

    with pytest.raises(PermissionError, match="authorization"):
        selector.fit(
            features=features,
            target=target,
            context=SelectorFitContext.TRAINING_SUPERVISED,
            target_access_authorized=False,
        )


def test_chi_square_selector_rejects_negative_features() -> None:
    selector = UnivariateSelector(
        task="classification",
        method=UnivariateMethod.CHI_SQUARE,
    )

    with pytest.raises(ValueError, match="nonnegative"):
        selector.fit(
            features=pd.DataFrame({"feature": [-1, 0, 1, 2]}),
            target=pd.Series([0, 0, 1, 1]),
            context=SelectorFitContext.TRAINING_SUPERVISED,
            target_access_authorized=True,
        )

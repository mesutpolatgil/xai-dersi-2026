import numpy as np
import pandas as pd

from model import FEATURES, ACTIONABLE


def find_counterfactual(
    model,
    instance: pd.DataFrame,
    reference: pd.DataFrame,
) -> pd.Series:

    desired = "good"

    predictions = model.predict(
        reference[FEATURES]
    )

    candidates = (
        reference.loc[
            predictions == desired,
            ACTIONABLE,
        ]
        .drop_duplicates()
    )

    if candidates.empty:
        raise RuntimeError(
            "Modelin good tahmini verdiği aday bulunamadı."
        )

    base = instance.iloc[0]

    trial = pd.DataFrame(
        [base] * len(candidates)
    ).reset_index(drop=True)

    trial[ACTIONABLE] = (
        candidates.reset_index(drop=True)
    )

    good_index = list(
        model.classes_
    ).index(desired)

    proba = model.predict_proba(
        trial[FEATURES]
    )[:, good_index]

    valid = (
        model.predict(trial[FEATURES])
        == desired
    )

    trial = trial.loc[valid].copy()
    proba = proba[valid]

    numeric_actionable = [
        "Credit amount",
        "Duration",
    ]

    ranges = (
        reference[numeric_actionable].max()
        - reference[numeric_actionable].min()
    ).replace(0, 1)

    distance = (
        (
            trial["Credit amount"]
            - base["Credit amount"]
        ).abs()
        / ranges["Credit amount"]

        +

        (
            trial["Duration"]
            - base["Duration"]
        ).abs()
        / ranges["Duration"]

        +

        (
            trial["Saving accounts"]
            != base["Saving accounts"]
        ).astype(float)

        +

        (
            trial["Checking account"]
            != base["Checking account"]
        ).astype(float)
    )

    best = int(
        np.lexsort(
            (
                -proba,
                distance.to_numpy(),
            )
        )[0]
    )

    result = trial.iloc[best].copy()

    result["P(good)"] = proba[best]

    return result
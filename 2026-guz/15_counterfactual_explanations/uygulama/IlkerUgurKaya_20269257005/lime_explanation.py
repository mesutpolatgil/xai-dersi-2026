import numpy as np
import pandas as pd

from model import FEATURES, CATEGORICAL


def lime_explanation(
    model,
    x_train: pd.DataFrame,
    instance: pd.DataFrame,
) -> list[tuple[str, float]]:

    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError as exc:
        raise RuntimeError(
            "LIME için `pip install lime` çalıştırın."
        ) from exc

    lime_train = x_train[FEATURES].copy()
    lime_instance = instance[FEATURES].copy()

    categorical_names: dict[int, list[str]] = {}
    category_values: dict[str, list[str]] = {}

    for column in CATEGORICAL:
        categories = sorted(
            x_train[column].astype(str).unique().tolist()
        )

        category_values[column] = categories

        position = FEATURES.index(column)
        categorical_names[position] = categories

        lookup = {
            value: i
            for i, value in enumerate(categories)
        }

        lime_train[column] = (
            lime_train[column]
            .astype(str)
            .map(lookup)
        )

        lime_instance[column] = (
            lime_instance[column]
            .astype(str)
            .map(lookup)
        )

    train_array = lime_train.to_numpy(dtype=float)
    instance_array = lime_instance.to_numpy(dtype=float)[0]

    def predict_from_lime(array: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(
            array,
            columns=FEATURES,
        )

        for column in CATEGORICAL:
            categories = category_values[column]

            codes = (
                np.rint(frame[column])
                .astype(int)
                .clip(0, len(categories) - 1)
            )

            frame[column] = [
                categories[code]
                for code in codes
            ]

        return model.predict_proba(frame[FEATURES])

    class_names = list(model.classes_)
    good_index = class_names.index("good")

    explainer = LimeTabularExplainer(
        train_array,
        feature_names=FEATURES,
        class_names=class_names,
        categorical_features=[
            FEATURES.index(column)
            for column in CATEGORICAL
        ],
        categorical_names=categorical_names,
        mode="classification",
        random_state=42,
    )

    explanation = explainer.explain_instance(
        instance_array,
        predict_from_lime,
        num_features=len(FEATURES),
        labels=[good_index],
    )

    return explanation.as_list(
        label=good_index
    )
#!/usr/bin/env python3

from pathlib import Path

import pandas as pd

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

from sklearn.model_selection import (
    train_test_split,
)

from model import (
    build_model,
    FEATURES,
    TARGET,
)

from lime_explanation import (
    lime_explanation,
)

from counterfactual_explanation import (
    find_counterfactual,
)


def select_method() -> str:
    print("\n" + "=" * 50)
    print("XAI YÖNTEMİ SEÇİMİ")
    print("=" * 50)

    print("1 - LIME")
    print("2 - Counterfactual")
    print("3 - Her ikisi")

    while True:
        choice = input("\nSeçiminiz: ").strip()

        if choice in {"1", "2", "3"}:
            return choice

        print(
            "Geçersiz seçim. "
            "1, 2 veya 3 giriniz."
        )


def run_lime(
    model,
    x_train,
    instance,
):
    print("\n" + "=" * 50)
    print("LIME AÇIKLAMASI")
    print("=" * 50)

    try:
        explanation = lime_explanation(
            model,
            x_train,
            instance,
        )

        print("\nGood sınıfına katkılar:\n")

        for rule, weight in explanation:
            print(
                f"{weight:+.4f}  {rule}"
            )

    except RuntimeError as exc:
        print("LIME çalıştırılamadı:", exc)


def run_counterfactual(
    model,
    x_train,
    instance,
):
    print("\n" + "=" * 50)
    print("COUNTERFACTUAL AÇIKLAMASI")
    print("=" * 50)

    cf = find_counterfactual(
        model,
        instance,
        x_train,
    )

    print(
        "\nDeğiştirilmesi önerilen özellikler:\n"
    )

    changed = False

    for column in FEATURES:
        old = instance.iloc[0][column]
        new = cf[column]

        if old != new:
            changed = True
            print(
                f"{column}: "
                f"{old} -> {new}"
            )

    if not changed:
        print("Özellik değişikliği bulunamadı.")

    print(
        f"\nYeni P(good): "
        f"{cf['P(good)']:.3f}"
    )


def main():
    csv_path = Path(
        input(
            "CSV dosyasının yolunu girin: "
        ).strip()
    )

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dosya bulunamadı: {csv_path}"
        )

    data = pd.read_csv(
        csv_path,
        keep_default_na=False,
    )

    x_train, x_test, y_train, y_test = (
        train_test_split(
            data[FEATURES],
            data[TARGET],
            test_size=0.25,
            stratify=data[TARGET],
            random_state=42,
        )
    )

    model = build_model()

    model.fit(
        x_train,
        y_train,
    )

    predictions = model.predict(x_test)

    print("\n" + "=" * 50)
    print("MODEL SONUÇLARI")
    print("=" * 50)

    print(
        classification_report(
            y_test,
            predictions,
            digits=3,
        )
    )

    print(
        "Confusion Matrix [bad, good]:"
    )

    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=["bad", "good"],
        )
    )

    # Test kümesinde modelin "bad"
    # tahmin ettiği ilk müşteriyi seç.
    bad_rows = x_test.loc[
        predictions == "bad"
    ]

    if bad_rows.empty:
        raise RuntimeError(
            "Test kümesinde bad tahmini "
            "yapılan müşteri bulunamadı."
        )

    row_id = bad_rows.index[0]

    instance = data.loc[
        [row_id],
        FEATURES,
    ]

    probabilities = dict(
        zip(
            model.classes_,
            model.predict_proba(instance)[0],
        )
    )

    print("\n" + "=" * 50)
    print("SEÇİLEN MÜŞTERİ")
    print("=" * 50)

    print(f"\nCSV satırı: {row_id}")

    print(
        instance.to_string(
            index=False
        )
    )

    print(
        "\nModel tahmini:",
        model.predict(instance)[0],
    )

    print(
        "Olasılıklar:",
        probabilities,
    )

    choice = select_method()

    if choice == "1":
        run_lime(
            model,
            x_train,
            instance,
        )

    elif choice == "2":
        run_counterfactual(
            model,
            x_train,
            instance,
        )

    elif choice == "3":
        run_lime(
            model,
            x_train,
            instance,
        )

        run_counterfactual(
            model,
            x_train,
            instance,
        )


if __name__ == "__main__":
    main()
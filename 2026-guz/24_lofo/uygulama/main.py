# -*- coding: utf-8 -*-
"""
Bölüm 24: Bir Özniteliği Dışarıda Bırakma (Leave One Feature Out - LOFO) Önemi
Yazar / Çeviri & Uygulama: Yunus Emre Yazarlı (20269258002)

Bu betik, Christoph Molnar'ın "Interpretable Machine Learning" kitabının 24. bölümündeki
tüm Random Forest modellerini 2/3 eğitim ve 1/3 test oranıyla veriden eğitir ve
bölümdeki 3 LOFO grafiğini üretir:
  1. fig-lofo-bike-1.png            (Bisiklet verisi - Random Forest LOFO)
  2. fig-lofo-bike-correlated-1.png (Bisiklet verisi + temp_copy - Korelasyonlu LOFO)
  3. fig-lofo-penguins-1.png        (Penguen verisi - Random Forest LOFO)
"""

import os
import tempfile
from pathlib import Path

# Matplotlib önbellek dizinini yazılabilir geçici klasöre yönlendir
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


# Klasör yolları (betiğin konumuna göre dinamik olarak çözülür)
SCRIPT_DIR = Path(__file__).resolve().parent
CHAPTER_DIR = SCRIPT_DIR.parent.parent
ROOT_DIR = CHAPTER_DIR.parent
DATA_DIR = ROOT_DIR / "data"

OUTPUT_DIRS = [
    CHAPTER_DIR / "yunusemreyazarli-20269258002_files" / "figure-html",
    CHAPTER_DIR / "lofo" / "figure-html",
]

BIKE_CAT_COLS = {"season", "holiday", "workday", "weather"}


def save_figure_to_all_dirs(fig: plt.Figure, filename: str):
    """Üretilen grafiği bölümün ilgili görsel klasörlerine kaydeder."""
    for out_dir in OUTPUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / filename, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fit_bike_rf(X_train: pd.DataFrame, y_train: pd.Series, seed: int = 42) -> Pipeline:
    """
    Orijinal R randomForest yapısına uygun olarak kategorik değişkenleri tek sütun
    halinde kodlar ve mtry = floor(p / 3) ile RandomForestRegressor eğitir.
    """
    cat_cols = [c for c in X_train.columns if c in BIKE_CAT_COLS]
    preprocessor = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols)],
        remainder="passthrough",
    )
    mtry = max(1, X_train.shape[1] // 3)
    model = Pipeline([
        ("prep", preprocessor),
        ("rf", RandomForestRegressor(n_estimators=300, max_features=mtry, random_state=seed)),
    ])
    model.fit(X_train, y_train)
    return model


def fit_penguins_rf(X_train: pd.DataFrame, y_train: pd.Series, seed: int = 42) -> Pipeline:
    """
    Penguen cinsiyet sınıflandırması için mtry = floor(sqrt(p)) ile
    RandomForestClassifier eğitir.
    """
    cat_cols = [c for c in X_train.columns if c == "species"]
    preprocessor = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols)],
        remainder="passthrough",
    )
    mtry = max(1, int(np.sqrt(X_train.shape[1])))
    model = Pipeline([
        ("prep", preprocessor),
        ("rf", RandomForestClassifier(n_estimators=300, max_features=mtry, random_state=seed)),
    ])
    model.fit(X_train, y_train)
    return model


def plot_lofo_bars(lofo_df: pd.DataFrame, xlabel: str, ylabel: str, filename: str):
    """LOFO önem değerlerini yatay sütun grafiği olarak çizer ve kaydeder."""
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8.2, 5.4))

    plot_df = lofo_df.sort_values("importance", ascending=True).reset_index(drop=True)
    y_pos = np.arange(len(plot_df))

    ax.barh(y_pos, plot_df["importance"], color="#595959", height=0.78, zorder=3)
    ax.axvline(0.0, color="#888888", linewidth=0.9, linestyle="-", zorder=2)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["feature"], fontsize=10.5)
    ax.set_xlabel(xlabel, fontsize=11.5)
    ax.set_ylabel(ylabel, fontsize=11.5)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color("#cccccc")
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#666666")

    fig.tight_layout()
    save_figure_to_all_dirs(fig, filename)


def run_bike_lofo():
    """1. ve 2. Analizler: Bisiklet verisinde standart LOFO ve temp_copy eklenmiş LOFO."""
    bike = pd.read_csv(DATA_DIR / "bike.csv")
    features = ["season", "holiday", "workday", "weather", "temp", "hum", "windspeed", "cnt_2d_bfr"]

    # Kitaptaki oranla 2/3 eğitim, 1/3 test ayrımı (aynı indeksler her iki deneyde kullanılır)
    train_idx, test_idx = train_test_split(bike.index, test_size=1 / 3, random_state=42)

    # --- 1. Standart Bisiklet Verisi LOFO ---
    print("=== 1. Bisiklet Verisi: Standart LOFO Öznitelik Önemi ===")
    X_train = bike.loc[train_idx, features]
    X_test = bike.loc[test_idx, features]
    y_train = bike.loc[train_idx, "cnt"]
    y_test = bike.loc[test_idx, "cnt"]

    base_model = fit_bike_rf(X_train, y_train, seed=42)
    base_mae = mean_absolute_error(y_test, base_model.predict(X_test))
    print(f"Test MAE (Tüm özniteliklerle): {base_mae:.2f}")

    records = []
    for idx, feat in enumerate(features):
        sub_model = fit_bike_rf(X_train.drop(columns=[feat]), y_train, seed=42 + idx + 1)
        new_mae = mean_absolute_error(y_test, sub_model.predict(X_test.drop(columns=[feat])))
        records.append({"feature": feat, "importance": float(new_mae - base_mae)})

    lofo_bike_df = pd.DataFrame(records).sort_values("importance", ascending=False).reset_index(drop=True)
    print(lofo_bike_df.to_string(index=False))

    plot_lofo_bars(
        lofo_df=lofo_bike_df,
        xlabel="LOFO Önemi (MAE farkı)",
        ylabel="Öznitelik",
        filename="fig-lofo-bike-1.png",
    )
    print("Kaydedildi: fig-lofo-bike-1.png\n")

    # --- 2. Korelasyonlu (temp + temp_copy) Bisiklet Verisi LOFO ---
    print("=== 2. Bisiklet Verisi: Korelasyonlu (temp_copy) LOFO Öznitelik Önemi ===")
    bike_corr = bike.copy()
    bike_corr["temp_copy"] = bike_corr["temp"]
    corr_features = features + ["temp_copy"]

    X_train_c = bike_corr.loc[train_idx, corr_features]
    X_test_c = bike_corr.loc[test_idx, corr_features]

    base_corr_model = fit_bike_rf(X_train_c, y_train, seed=42)
    base_corr_mae = mean_absolute_error(y_test, base_corr_model.predict(X_test_c))
    print(f"Test MAE (temp_copy dahil tüm özniteliklerle): {base_corr_mae:.2f}")

    corr_records = []
    for idx, feat in enumerate(corr_features):
        sub_model = fit_bike_rf(X_train_c.drop(columns=[feat]), y_train, seed=42 + idx + 1)
        new_mae = mean_absolute_error(y_test, sub_model.predict(X_test_c.drop(columns=[feat])))
        corr_records.append({"feature": feat, "importance": float(new_mae - base_corr_mae)})

    lofo_corr_df = pd.DataFrame(corr_records).sort_values("importance", ascending=False).reset_index(drop=True)
    print(lofo_corr_df.to_string(index=False))

    plot_lofo_bars(
        lofo_df=lofo_corr_df,
        xlabel="LOFO Önemi (MAE farkı)",
        ylabel="Öznitelik",
        filename="fig-lofo-bike-correlated-1.png",
    )
    print("Kaydedildi: fig-lofo-bike-correlated-1.png\n")


def run_penguins_lofo():
    """3. Analiz: Penguen cinsiyet sınıflandırması Random Forest LOFO."""
    print("=== 3. Penguen Verisi: Random Forest LOFO Öznitelik Önemi ===")
    penguins = pd.read_csv(DATA_DIR / "penguins.csv")
    features = ["species", "bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    penguins = penguins[features + ["sex"]].dropna().reset_index(drop=True)

    X = penguins[features]
    y = penguins["sex"]

    # Kitaptaki oranla 2/3 eğitim, 1/3 test ayrımı
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=4)

    base_model = fit_penguins_rf(X_train, y_train, seed=4)
    base_acc = accuracy_score(y_test, base_model.predict(X_test))
    print(f"Test Doğruluk (Tüm özniteliklerle): {base_acc:.4f}")

    records = []
    for idx, feat in enumerate(features):
        sub_model = fit_penguins_rf(X_train.drop(columns=[feat]), y_train, seed=4 + idx + 1)
        new_acc = accuracy_score(y_test, sub_model.predict(X_test.drop(columns=[feat])))
        # Doğrulukta büyük değer daha iyi olduğundan fark -1 ile çarpılır: base_acc - new_acc
        records.append({"feature": feat, "importance": float(base_acc - new_acc)})

    lofo_peng_df = pd.DataFrame(records).sort_values("importance", ascending=False).reset_index(drop=True)
    print(lofo_peng_df.to_string(index=False))

    plot_lofo_bars(
        lofo_df=lofo_peng_df,
        xlabel="LOFO Önemi (doğruluk farkı)",
        ylabel="",
        filename="fig-lofo-penguins-1.png",
    )
    print("Kaydedildi: fig-lofo-penguins-1.png\n")


def main():
    run_bike_lofo()
    run_penguins_lofo()
    print("Bölüm 24 analizleri ve 3 grafik üretimi başarıyla tamamlandı.")


if __name__ == "__main__":
    main()

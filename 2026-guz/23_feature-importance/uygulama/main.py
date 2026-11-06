# -*- coding: utf-8 -*-
"""
Bölüm 23: Permütasyon Öznitelik Önemi (Permutation Feature Importance - PFI)
Yazar / Çeviri & Uygulama: Yunus Emre Yazarlı (20269258002)

Bu betik, Christoph Molnar'ın "Interpretable Machine Learning" kitabının 23. bölümündeki
tüm modelleri 2/3 eğitim ve 1/3 test oranıyla veriden eğitir ve bölümdeki 4 grafiği üretir:
  1. fig-importance-bike-1.png               (Bisiklet verisi - SVM modeli PFI)
  2. fig-importance-penguins-1.png           (Penguen verisi - 3 Lojistik Regresyon PFI)
  3. fig-cor-by-species-1.png                (Penguen verisi - Türe göre korelasyon)
  4. fig-importance-penguins-by-species-1.png (Penguen verisi - Grup bazında PFI)
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
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR


# Klasör yolları (betiğin konumuna göre dinamik olarak çözülür)
SCRIPT_DIR = Path(__file__).resolve().parent
CHAPTER_DIR = SCRIPT_DIR.parent.parent
ROOT_DIR = CHAPTER_DIR.parent
DATA_DIR = ROOT_DIR / "data"

OUTPUT_DIRS = [
    CHAPTER_DIR / "yunusemreyazarli-20269258002_files" / "figure-html",
    CHAPTER_DIR / "feature-importance" / "figure-html",
]


class SpeciesLogisticEnsemble:
    """
    Kitaptaki 'pengu_logreg' yapısını birebir uygular:
    Her penguen türü (Adelie, Chinstrap, Gentoo) için ayrı bir lojistik regresyon
    modeli eğitir ve tahmin aşamasında tüm sistemi tek bir kara kutu (black-box)
    model olarak ele alarak örnekleri 'species' sütununa göre ilgili modele yönlendirir.
    """

    def __init__(self, species_list, numeric_features):
        self.species_list = species_list
        self.numeric_features = numeric_features
        self.models = {}

    def fit(self, X: pd.DataFrame, y: pd.Series):
        for sp in self.species_list:
            mask = (X["species"] == sp).values
            clf = LogisticRegression(C=1e6, max_iter=5000, random_state=42)
            clf.fit(X.loc[mask, self.numeric_features], y.loc[mask])
            self.models[sp] = clf
        return self

    def predict_proba_female(self, X: pd.DataFrame) -> np.ndarray:
        probs = np.zeros(len(X), dtype=float)
        for sp, clf in self.models.items():
            mask = (X["species"] == sp).values
            if np.any(mask):
                probs[mask] = clf.predict_proba(X.loc[mask, self.numeric_features])[:, 1]
        return np.clip(probs, 1e-15, 1.0 - 1e-15)


def save_figure_to_all_dirs(fig: plt.Figure, filename: str):
    """Üretilen grafiği bölümün ilgili görsel klasörlerine kaydeder."""
    for out_dir in OUTPUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / filename, dpi=160, bbox_inches="tight")
    plt.close(fig)


def compute_pfi_ratio(X_eval: pd.DataFrame, y_eval, loss_fn, features: list, n_repeats: int = 50, seed: int = 42) -> pd.DataFrame:
    """
    Tekrarlı permütasyon ile oran tabanlı PFI (e_perm / e_orig) hesaplar.
    Ortalama önem ile %5 ve %95 kantil aralığını döndürür.
    """
    rng = np.random.default_rng(seed)
    base_loss = loss_fn(X_eval, y_eval)
    records = []

    for feat in features:
        ratios = []
        col_values = X_eval[feat].to_numpy()
        for _ in range(n_repeats):
            X_perm = X_eval.copy()
            X_perm[feat] = rng.permutation(col_values)
            perm_loss = loss_fn(X_perm, y_eval)
            ratios.append(perm_loss / base_loss)

        records.append({
            "feature": feat,
            "importance_mean": float(np.mean(ratios)),
            "q05": float(np.quantile(ratios, 0.05)),
            "q95": float(np.quantile(ratios, 0.95)),
        })

    return pd.DataFrame(records).sort_values("importance_mean", ascending=False).reset_index(drop=True)


def run_bike_svm_pfi():
    """1. Analiz: Bisiklet kiralama verisinde SVM modeli ve PFI grafiği."""
    print("=== 1. Bisiklet Verisi: SVM Permütasyon Öznitelik Önemi ===")
    bike = pd.read_csv(DATA_DIR / "bike.csv")
    features = ["season", "holiday", "workday", "weather", "temp", "hum", "windspeed", "cnt_2d_bfr"]
    cat_cols = ["season", "holiday", "workday", "weather"]

    X = bike[features]
    y = bike["cnt"]

    # Kitaptaki oranla 2/3 eğitim, 1/3 test ayrımı
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=42)

    svm_pipeline = Pipeline([
        ("ohe", ColumnTransformer([("cat", OneHotEncoder(drop="first", sparse_output=False), cat_cols)], remainder="passthrough")),
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf", C=1.0, epsilon=0.1)),
    ])
    model = TransformedTargetRegressor(regressor=svm_pipeline, transformer=StandardScaler())
    model.fit(X_train, y_train)

    base_mae = mean_absolute_error(y_test, model.predict(X_test))
    print(f"Test MAE (Orijinal): {base_mae:.2f}")

    pfi_df = compute_pfi_ratio(
        X_eval=X_test,
        y_eval=y_test,
        loss_fn=lambda X_df, y_true: mean_absolute_error(y_true, model.predict(X_df)),
        features=features,
        n_repeats=50,
        seed=42,
    )
    print(pfi_df.to_string(index=False))

    # Görselleştirme: fig-importance-bike-1.png
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8.2, 5.4))

    plot_df = pfi_df.iloc[::-1].reset_index(drop=True)
    y_pos = np.arange(len(plot_df))

    ax.hlines(y=y_pos, xmin=plot_df["q05"], xmax=plot_df["q95"], color="#284b4b", linewidth=3.5, zorder=2)
    ax.scatter(plot_df["importance_mean"], y_pos, color="black", s=55, zorder=3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["feature"], fontsize=10.5)
    ax.set_xlim(left=0.95)
    ax.set_xlabel("Öznitelik önemi (MAE oranı)", fontsize=11.5)
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=10)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color("#cccccc")
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#666666")

    fig.tight_layout()
    save_figure_to_all_dirs(fig, "fig-importance-bike-1.png")
    print("Kaydedildi: fig-importance-bike-1.png\n")


def run_penguins_pfi_and_correlations():
    """2, 3 ve 4. Analizler: Penguen verisi PFI, korelasyon ve grup bazında PFI grafikleri."""
    print("=== 2. Penguen Verisi: 3 Lojistik Regresyon PFI ===")
    penguins_raw = pd.read_csv(DATA_DIR / "penguins.csv")
    features = ["species", "bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    numeric_features = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    species_order = ["Adelie", "Chinstrap", "Gentoo"]

    penguins_clean = penguins_raw[features + ["sex"]].dropna().reset_index(drop=True)
    X = penguins_clean[features]
    y = (penguins_clean["sex"] == "female").astype(int)

    # Kitaptaki oranla 2/3 eğitim, 1/3 test ayrımı
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=10)

    ensemble = SpeciesLogisticEnsemble(species_list=species_order, numeric_features=numeric_features)
    ensemble.fit(X_train, y_train)

    loss_fn = lambda X_df, y_true: log_loss(y_true, ensemble.predict_proba_female(X_df), labels=[0, 1])
    base_ll = loss_fn(X_test, y_test)
    print(f"Test Log-Loss (Orijinal): {base_ll:.4f}")

    pfi_penguins = compute_pfi_ratio(
        X_eval=X_test,
        y_eval=y_test,
        loss_fn=loss_fn,
        features=features,
        n_repeats=50,
        seed=42,
    )
    print(pfi_penguins.to_string(index=False))

    # Görselleştirme 2: fig-importance-penguins-1.png
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8.2, 5.4))

    plot_df = pfi_penguins.iloc[::-1].reset_index(drop=True)
    y_pos = np.arange(len(plot_df))

    ax.hlines(y=y_pos, xmin=plot_df["q05"], xmax=plot_df["q95"], color="#284b4b", linewidth=3.5, zorder=2)
    ax.scatter(plot_df["importance_mean"], y_pos, color="black", s=55, zorder=3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["feature"], fontsize=10.5)
    ax.set_xlim(left=0.8)
    ax.set_xlabel("Öznitelik önemi (log kaybı oranı)", fontsize=11.5)
    ax.set_ylabel("")

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color("#cccccc")
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#666666")

    fig.tight_layout()
    save_figure_to_all_dirs(fig, "fig-importance-penguins-1.png")
    print("Kaydedildi: fig-importance-penguins-1.png\n")

    # === 3. Korelasyon Grafiği: fig-cor-by-species-1.png ===
    print("=== 3. Penguen Verisi: Türe Göre Korelasyon Analizi ===")
    cor_df = penguins_raw[["species", "flipper_length_mm", "body_mass_g"]].dropna().reset_index(drop=True)
    palette = {"Adelie": "#440154", "Chinstrap": "#21918c", "Gentoo": "#fde725"}

    panels = [
        ("Adelie", cor_df[cor_df["species"] == "Adelie"]),
        ("all", cor_df),
        ("Chinstrap", cor_df[cor_df["species"] == "Chinstrap"]),
        ("Gentoo", cor_df[cor_df["species"] == "Gentoo"]),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4))
    for ax, (title, sub) in zip(axes.flat, panels):
        for sp in species_order:
            sp_rows = sub[sub["species"] == sp]
            if not sp_rows.empty:
                ax.scatter(
                    sp_rows["flipper_length_mm"],
                    sp_rows["body_mass_g"],
                    color=palette[sp],
                    edgecolors="#333333" if sp == "Gentoo" else "none",
                    linewidths=0.4 if sp == "Gentoo" else 0.0,
                    s=28,
                    alpha=0.9,
                    label=sp,
                )
        r_val = sub["flipper_length_mm"].corr(sub["body_mass_g"])
        print(f"  Alt küme '{title}': Pearson r = {r_val:.2f}")
        ax.text(
            0.05,
            0.90,
            f"r = {r_val:.2f}",
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#333333", alpha=0.95),
        )
        ax.set_title(title, fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.supxlabel("Yüzgeç uzunluğu (mm)", fontsize=12, y=0.02)
    fig.supylabel("Vücut kütlesi (gram)", fontsize=12, x=0.02)

    handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w", label=sp,
            markerfacecolor=palette[sp],
            markeredgecolor="#333333" if sp == "Gentoo" else palette[sp],
            markersize=7,
        )
        for sp in species_order
    ]
    fig.legend(handles=handles, title="species", loc="center right", bbox_to_anchor=(0.99, 0.5), frameon=False)
    fig.tight_layout(rect=[0.03, 0.04, 0.87, 1.0])
    save_figure_to_all_dirs(fig, "fig-cor-by-species-1.png")
    print("Kaydedildi: fig-cor-by-species-1.png\n")

    # === 4. Grup Bazında PFI: fig-importance-penguins-by-species-1.png ===
    print("=== 4. Penguen Verisi: Grup Bazında (Türe Göre) PFI ===")
    group_rows = []
    for sp in species_order:
        sub_mask = (X_test["species"] == sp).values
        X_sub = X_test.loc[sub_mask].reset_index(drop=True)
        y_sub = y_test.loc[sub_mask].reset_index(drop=True)

        sp_pfi = compute_pfi_ratio(
            X_eval=X_sub,
            y_eval=y_sub,
            loss_fn=loss_fn,
            features=features,
            n_repeats=50,
            seed=42,
        )
        sp_pfi["species"] = sp
        group_rows.append(sp_pfi)

    group_df = pd.concat(group_rows, ignore_index=True)
    print(group_df[["species", "feature", "importance_mean"]].to_string(index=False))

    feature_y_order = ["bill_depth_mm", "bill_length_mm", "body_mass_g", "flipper_length_mm", "species"]
    y_map = {f: idx for idx, f in enumerate(feature_y_order)}
    offsets = {"Adelie": 0.14, "Chinstrap": 0.0, "Gentoo": -0.14}
    markers = {"Adelie": "o", "Chinstrap": "^", "Gentoo": "s"}

    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    for sp in species_order:
        sub = group_df[group_df["species"] == sp]
        ys = [y_map[f] + offsets[sp] for f in sub["feature"]]
        ax.scatter(
            sub["importance_mean"],
            ys,
            color=palette[sp],
            marker=markers[sp],
            edgecolors="#333333" if sp == "Gentoo" else "none",
            linewidths=0.6 if sp == "Gentoo" else 0.0,
            s=75,
            label=sp,
            zorder=3,
        )

    ax.set_yticks(range(len(feature_y_order)))
    ax.set_yticklabels(feature_y_order, fontsize=10.5)
    ax.set_xlabel("Öznitelik önemi (log kaybı oranı)", fontsize=11.5)
    ax.set_ylabel("Öznitelik", fontsize=11.5)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color("#cccccc")
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#666666")

    ax.legend(title="species", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    save_figure_to_all_dirs(fig, "fig-importance-penguins-by-species-1.png")
    print("Kaydedildi: fig-importance-penguins-by-species-1.png\n")


def main():
    run_bike_svm_pfi()
    run_penguins_pfi_and_correlations()
    print("Bölüm 23 analizleri ve 4 grafik üretimi başarıyla tamamlandı.")


if __name__ == "__main__":
    main()

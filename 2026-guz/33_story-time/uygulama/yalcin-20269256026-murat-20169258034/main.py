"""Bölüm 33 ve 34 için kontrollü kısayol öğrenmesi deneyi.

Gerçek UCI Steel Plates Faults ölçümleri + açıkça yapay istasyon kodu.
Bu dosya bir makalenin yeniden üretimi değildir. Kod lisansı MIT.
Veri lisansı CC BY 4.0; ayrıntılar README.md içindedir.
"""
# %% imports
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split

SEED = 42
STATION = "Yapay_istasyon_kodu"

# %% data
def load_data(data_dir):
    """27 gerçek girdi ve 7 ayrı hedef sütununu ayırır."""
    names = (data_dir / "Faults27x7_var").read_text().splitlines()
    matrix = np.loadtxt(data_dir / "Faults.NNA")
    assert matrix.shape == (1941, 34), matrix.shape
    labels = matrix[:, 27:]
    assert np.all(labels.sum(axis=1) == 1), "Hedefler one-hot olmalı."
    return matrix[:, :27], labels.argmax(axis=1), names[:27], names[27:]


def station_code(y, seed, correlated=True):
    """KONTROLLÜ SIZINTI: etiket kullanılarak oluşturulan yapay özellik.

    Gerçek veri kümesinde böyle bir istasyon bilgisi yoktur. Bu özellik
    gerçek üretimde kullanılabilir bir girdi olarak sunulmamaktadır.
    """
    rng = np.random.default_rng(seed)
    if correlated:
        code = y.copy()
        flip = rng.random(len(y)) < 0.02
        # Değiştirilen kod mutlaka başka bir sınıfı gösterir.
        code[flip] = (code[flip] + rng.integers(1, 7, flip.sum())) % 7
        return code
    # Müdahale senaryosu: istasyon kodu etiketten bağımsızdır.
    return rng.integers(0, 7, len(y))


def metrics(y, prediction):
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro")),
    }

# %% experiment
def run_experiment(data_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    X, y, names, classes = load_data(data_dir)
    train, test = train_test_split(
        np.arange(len(y)), test_size=0.30, stratify=y, random_state=SEED
    )
    # Önce bölme yapılır; müdahale test örneklerinin fiziksel girdilerini
    # ve etiketlerini değiştirmez. Bütün modeller aynı satırlarda sınanır.
    X_train = np.column_stack([X[train], station_code(y[train], 100)])
    X_id = np.column_stack([X[test], station_code(y[test], 101)])
    X_shift = np.column_stack([X[test], station_code(y[test], 102, False)])
    y_test = y[test]
    config = dict(n_estimators=120, max_depth=7, min_samples_leaf=3,
                  max_features=None, random_state=SEED, n_jobs=1)
    shortcut = RandomForestClassifier(**config).fit(X_train, y[train])
    without_station = RandomForestClassifier(**config).fit(X[train], y[train])
    result = {
        "seed": SEED,
        "rows_train": len(train), "rows_test": len(test),
        "dataset_sha256": hashlib.sha256((data_dir / "Faults.NNA").read_bytes()).hexdigest(),
        "versions": {"numpy": np.__version__, "sklearn": sklearn.__version__,
                     "matplotlib": matplotlib.__version__},
        "synthetic_feature": STATION,
        "models": {},
    }
    for key, model, a, b in [
        ("istasyon_dahil", shortcut, X_id, X_shift),
        ("istasyon_haric", without_station, X[test], X[test]),
    ]:
        result["models"][key] = {
            "ayni_korelasyon": metrics(y_test, model.predict(a)),
            "istasyon_bagimsiz": metrics(y_test, model.predict(b)),
        }
    # Aynı eğitilmiş modele aynı girdi yeniden verildiğinde aynı tahmin.
    assert np.array_equal(shortcut.predict(X_shift), shortcut.predict(X_shift))
    # Aynı dağılımdan ayrılmış test ve müdahale senaryosu ayrı yorumlanır.
    importances = {}
    for condition, data in [("ayni_korelasyon", X_id), ("istasyon_bagimsiz", X_shift)]:
        pfi = permutation_importance(
            shortcut, data, y_test, scoring="balanced_accuracy",
            n_repeats=8, random_state=SEED, n_jobs=1,
        )
        importances[condition] = {
            feature: {"mean": float(mean), "std": float(std)}
            for feature, mean, std in zip(names + [STATION], pfi.importances_mean, pfi.importances_std)
        }
    result["permutation_importance"] = importances

    # Bir parçanın gerçek ölçümleri sabit kalırken yalnızca istasyon
    # kodunu değiştiririz. Bu, nedensel bir fizik deneyi değildir.
    variants = np.repeat(X_id[:1], 7, axis=0)
    variants[:, -1] = np.arange(7)
    result["single_instance"] = {
        "original_row": int(test[0]), "true_class": classes[y_test[0]],
        "predictions_by_station": [classes[i] for i in shortcut.predict(variants)],
    }
    (output_dir / "sonuclar.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
    xx = np.arange(2)
    for offset, (key, label, color) in zip([-0.18, 0.18], [
        ("istasyon_dahil", "Yapay istasyon dahil", "#b74c3c"),
        ("istasyon_haric", "Yalnız gerçek ölçümler", "#246d8f"),
    ]):
        scores = [result["models"][key][c]["balanced_accuracy"]
                  for c in ["ayni_korelasyon", "istasyon_bagimsiz"]]
        bars = ax.bar(xx + offset, scores, width=.34, label=label, color=color)
        ax.bar_label(bars, labels=[f"{v:.1%}" for v in scores], padding=4)
    ax.set(xticks=xx, xticklabels=["Aynı yapay korelasyon", "İstasyon etiketten bağımsız"],
           ylim=(0, 1.1), ylabel="Dengeli doğruluk", title="Aynı test parçalarında kontrollü müdahale")
    ax.legend(loc="upper right", frameon=False)
    fig.savefig(output_dir / "performans.png", dpi=160)
    plt.close(fig)
    top = sorted(names + [STATION], key=lambda f: importances["ayni_korelasyon"][f]["mean"])[-7:]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained", sharey=True)
    for ax, (condition, title) in zip(axes, [
        ("ayni_korelasyon", "Aynı yapay korelasyon"),
        ("istasyon_bagimsiz", "İstasyon etiketten bağımsız"),
    ]):
        means = [importances[condition][f]["mean"] for f in top]
        stds = [importances[condition][f]["std"] for f in top]
        ax.barh(top, means, xerr=stds, color=["#b74c3c" if f == STATION else "#246d8f" for f in top])
        ax.axvline(0, color="#666666", linewidth=.8)
        ax.set(title=title, xlabel="Permütasyon sonrası skor düşüşü")
    fig.savefig(output_dir / "ozellik_onemi.png", dpi=160)
    plt.close(fig)
    print("Gerçek UCI ölçümleri; yapay istasyon kodu kontrollü olarak eklendi.")
    for model, conditions in result["models"].items():
        for condition, scores in conditions.items():
            print(f"{model:18} | {condition:20} | dengeli doğruluk {scores['balanced_accuracy']:.3f} | makro F1 {scores['macro_f1']:.3f}")
    print("Sabit fiziksel ölçümler, değişen istasyon kodları:", result["single_instance"])
    return result

# %% cli
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    run_experiment(args.data_dir, args.output_dir)

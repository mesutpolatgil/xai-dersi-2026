"""Check the Chapter 3 reproduction artifacts without retraining."""
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
import json
import re

import mistune
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parent
METHODS = {
    "Gradient", "Gradient × Input", "Integrated Gradients", "SmoothGrad",
    "Guided Backprop", "Grad-CAM", "Guided Grad-CAM",
}


def read_csv(name):
    return pd.read_csv(ROOT / name, encoding="utf-8-sig")


def verify_images():
    manifest = read_csv("veri.csv")
    expected = {
        ("synthetic", "train"): 256, ("synthetic", "validation"): 64,
        ("synthetic", "test"): 96, ("spurious", "train"): 256,
        ("spurious", "test"): 96, ("counterfactual", "test"): 96,
        ("background_only", "test"): 96, ("ood", "test"): 96,
    }
    assert manifest.groupby(["kind", "split"]).size().to_dict() == expected
    assert manifest.image_id.is_unique and manifest.path.is_unique
    assert manifest.label.isin([0, 1]).all() and manifest.cue.isin([0, 1]).all()
    assert manifest.class_name.equals(manifest.label.map({0: "kuş simgesi", 1: "köpek simgesi"}))
    for key, size in expected.items():
        kind, split = key
        counts = manifest[(manifest.kind == kind) & (manifest.split == split)].label.value_counts()
        assert counts.to_dict() == {0: size // 2, 1: size // 2}, key
    seeds = manifest.groupby("seed").agg(splits=("split", "nunique"), labels=("label", "nunique"))
    assert len(seeds) == 416 and (seeds == 1).all().all(), "Bölme veya etiket tohum çakışması"
    pixel_splits = defaultdict(set)
    for row in manifest.itertuples():
        relative_path = f"data/{row.kind}/{row.split}/{row.image_id}.png"
        assert row.path == relative_path, row.path
        file = (ROOT / row.path).resolve()
        assert file.is_relative_to(ROOT / "data") and file.is_file(), file
        assert sha256(file.read_bytes()).hexdigest() == row.sha256, file
        with Image.open(file) as image:
            assert (image.format, image.mode, image.size) == ("PNG", "L", (32, 32)), file
            pixel_splits[sha256(image.tobytes()).hexdigest()].add(row.split)
    assert all(len(splits) == 1 for splits in pixel_splits.values()), "Bölmeler arası piksel tekrarı"
    on_disk = {path.relative_to(ROOT).as_posix() for path in (ROOT / "data").rglob("*.png")
               if path.relative_to(ROOT / "data").parts[0] != "raw"}
    assert on_disk == set(manifest.path), "Disk ve manifesto PNG listeleri farklı"
    assert (manifest.loc[manifest.kind == "counterfactual", "label"] !=
            manifest.loc[manifest.kind == "counterfactual", "cue"]).all()
    for group in (manifest[manifest.kind == kind] for kind in ("background_only", "spurious")):
        test = group[group.split == "test"]
        assert (test.label == test.cue).all()
    return manifest


def tr(value, places=3):
    return "tanımsız" if pd.isna(value) else f"{value:.{places}f}".replace(".", ",")


def verify_measurements(manifest, markdown):
    scores = read_csv("model_scores.csv")
    assert set(scores.model) == {"normal", "spurious", "wrong_label"}
    assert set(scores.dataset) == {
        "independent_test", "correlated_test", "reversed_background", "background_only", "ood",
    }
    assert len(scores) == 15 and not scores.duplicated(["model", "dataset"]).any()
    assert scores.accuracy.between(0, 1).all() and scores.cross_entropy.ge(0).all()
    shown = scores.assign(accuracy=scores.accuracy.map(tr), cross_entropy=scores.cross_entropy.map(tr))
    shown = shown.rename(columns={"model": "Model", "dataset": "Test koşulu",
                                   "accuracy": "Doğruluk", "cross_entropy": "Çapraz entropi"})
    assert shown.to_markdown(index=False) in markdown

    images = manifest.set_index("image_id")
    bg = read_csv("background_attribution.csv")
    assert len(bg) == 224 and not bg.duplicated(["image_id", "model", "method"]).any()
    assert set(bg.model) == {"normal", "spurious"} and set(bg.method) == METHODS
    # Float32 numerator and denominator can differ by one rounding unit.
    assert bg.abs_bg_fraction.between(0, 1 + 2e-7).all()
    assert bg.image_id.nunique() == 16 and bg.groupby("image_id").size().eq(14).all()
    linked = images.loc[bg.image_id.unique()]
    assert linked.kind.eq("spurious").all() and linked.split.eq("test").all()

    similarity = read_csv("benzerlik_sonuclari.csv")
    expected = {"ood_proxy": 112, "reinitialized_weights": 448, "wrong_label": 112}
    assert similarity.experiment.value_counts().to_dict() == expected
    assert not similarity.duplicated(["experiment", "image_id", "method", "stage"]).any()
    assert similarity.ssim.between(-1, 1).all()
    for metric in ("spearman", "spearman_abs"):
        assert similarity[metric].dropna().between(-1 - 1e-12, 1 + 1e-12).all()
    reinit = similarity[similarity.experiment == "reinitialized_weights"]
    stages = {"head": 1, "fc": 2, "conv2": 3, "conv1": 4}
    assert reinit.stage.map(stages).eq(reinit.n_layers_reset).all()
    assert similarity.loc[similarity.experiment != "reinitialized_weights", "stage"].isna().all()
    expected_ids = {"wrong_label": "train", "reinitialized_weights": "test", "ood_proxy": "test"}
    for experiment, part in similarity.groupby("experiment"):
        assert set(part.method) == METHODS and part.image_id.nunique() == 16
        refs = images.loc[part.image_id.unique()]
        assert refs.kind.eq("synthetic").all() and refs.split.eq(expected_ids[experiment]).all()

    for experiment in ("wrong_label", "reinitialized_weights", "ood_proxy"):
        part = similarity[similarity.experiment == experiment]
        keys = ["stage", "method"] if experiment == "reinitialized_weights" else ["method"]
        table = part.groupby(keys, as_index=False)[["spearman", "ssim"]].mean()
        for key in ("spearman", "ssim"):
            table[key] = table[key].map(tr)
        table = table.rename(columns={"stage": "Son sıfırlanan katman", "method": "Yöntem",
                                      "spearman": "Spearman", "ssim": "SSIM"})
        assert table.to_markdown(index=False) in markdown, experiment

    report_rows = []
    for (experiment, method), chunk in similarity.groupby(["experiment", "method"]):
        options = sorted(chunk.stage.dropna().unique()) if experiment == "reinitialized_weights" else ["—"]
        for stage in options:
            sub = chunk if stage == "—" else chunk[chunk.stage == stage]
            assert len(sub) == 16
            for metric in ("spearman", "spearman_abs", "ssim"):
                valid = sub[metric].dropna()
                report_rows.append({
                    "Deney": experiment, "Aşama": stage, "Yöntem": method, "Ölçüt": metric,
                    "Ortalama": tr(valid.mean()), "Örnek ss": tr(valid.std(ddof=1)),
                    "Tanımlı çift": len(valid), "Toplam çift": len(sub),
                })
    assert pd.DataFrame(report_rows).to_markdown(index=False) in markdown
    return len(similarity)


def main():
    manifest = verify_images()
    markdown = (ROOT / "makale.md").read_text(encoding="utf-8")
    html = (ROOT / "makale.html").read_text(encoding="utf-8")
    body = mistune.create_markdown(plugins=["table"])(markdown)
    assert html.split("<main>", 1)[1].split("</main>", 1)[0] == body
    figures = sorted((ROOT / "sekiller").glob("*.png"))
    links = re.findall(r"!\[[^\]]*\]\((sekiller/[^)]+)\)", markdown)
    assert [f.stem[:2] for f in figures] == [f"{i:02d}" for i in range(1, 11)]
    assert links == [file.relative_to(ROOT).as_posix() for file in figures]
    for file in figures:
        with Image.open(file) as figure:
            assert figure.format == "PNG", file
            figure.verify()
    notebook = json.loads((ROOT / "analiz.ipynb").read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell["id"], "exec")
            assert all(result["output_type"] != "error" for result in cell["outputs"]), cell["id"]
    n = verify_measurements(manifest, markdown)
    counts = Counter(manifest.sha256)
    print(f"BAŞARILI: {len(manifest)} PNG, {len(figures)} şekil, {n} benzerlik ölçümü. "
          f"{sum(count - 1 for count in counts.values())} bölme içi PNG tekrarı; "
          "bölmeler arası piksel tekrarı yok. CSV ve rapor özetleri eşleşti.")


if __name__ == "__main__":
    main()

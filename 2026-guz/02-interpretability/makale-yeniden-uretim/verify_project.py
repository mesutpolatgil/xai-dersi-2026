"""Verify the saved Chapter 2 synthetic run without executing notebook code."""
import ast
import base64
from collections import Counter
import csv
from decimal import Decimal
from hashlib import sha256
from html import unescape
from io import BytesIO
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
SPLITS = {"train": 256, "validation": 64, "test": 96}
CLASSES = {0: "halka", 1: "arti"}
METHODS = (
    "Gradient", "Gradient × Input", "Integrated Gradients", "SmoothGrad",
    "Guided Backprop", "Grad-CAM", "Guided Grad-CAM",
)
FIGURES = (
    "01-veri.png", "02-egitim.png", "03-hatalar.png", "04-yontemler.png",
    "05-kademeli.png", "06-bagimsiz.png", "07-etiket.png", "08-ezber.png",
    "09-benzerlik.png", "10-zafiyet.png",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_dataset():
    with (ROOT / "veri.csv").open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    counts = Counter((row["split"], int(row["label"])) for row in rows)
    expected = {(split, label): size // 2 for split, size in SPLITS.items() for label in CLASSES}
    require(counts == expected, f"Beklenmeyen bölme/sınıf sayıları: {counts}")
    ids, paths, seeds, hashes, pixels = set(), set(), set(), set(), set()
    for row in rows:
        image_id, split, label = row["image_id"], row["split"], int(row["label"])
        relative = f"data/synthetic/{split}/{image_id}.png"
        prefix, _, suffix = image_id.rpartition("-")
        require(prefix == split and suffix.isdecimal() and int(suffix) % 2 == label,
                f"Görüntü kimliği, sınıf veya bölme uyuşmuyor: {image_id}")
        require(row["path"] == relative and row["group_id"] == image_id
                and row["class_name"] == CLASSES[label] and row["provenance"] == "synthetic",
                f"Manifesto bilgisi uyuşmuyor: {image_id}")
        require(image_id not in ids and relative not in paths and row["seed"] not in seeds,
                f"Yinelenen kimlik, yol veya tohum: {image_id}")
        file = (ROOT / relative).resolve()
        require(file.is_relative_to(ROOT / "data" / "synthetic") and file.is_file(),
                f"Görüntü dosyası eksik veya kapsam dışı: {relative}")
        digest = sha256(file.read_bytes()).hexdigest()
        require(digest == row["sha256"] and digest not in hashes, f"SHA-256 hatası: {relative}")
        with Image.open(file) as image:
            require((image.format, image.mode, image.size) == ("PNG", "L", (32, 32)),
                    f"Beklenmeyen görüntü biçimi: {relative}")
            pixel_digest = sha256(image.tobytes()).hexdigest()
        require(pixel_digest not in pixels, f"Birebir piksel tekrarı: {relative}")
        ids.add(image_id)
        paths.add(relative)
        seeds.add(row["seed"])
        hashes.add(digest)
        pixels.add(pixel_digest)
    found = {path.relative_to(ROOT).as_posix() for path in (ROOT / "data" / "synthetic").rglob("*.png")}
    require(found == paths, "Diskteki sentetik PNG listesi veri.csv ile eşleşmiyor.")
    return len(rows)


def verify_notebook():
    notebook = json.loads((ROOT / "analiz.ipynb").read_text(encoding="utf-8"))
    require(notebook["nbformat"] == 4, "Beklenmeyen Jupyter dosya biçimi.")
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    require(len(cells) == 9 and [c["execution_count"] for c in cells] == list(range(1, 10)),
            "Kaydedilmiş kod hücreleri sıralı yürütme bilgisi içermiyor.")
    figures, streams = [], []
    for cell in cells:
        source = "".join(cell["source"])
        compile(source, cell["id"], "exec")
        for output in cell["outputs"]:
            require(output["output_type"] != "error", f"Kaydedilmiş hücre hatası: {cell['id']}")
            if output["output_type"] == "stream":
                streams.append("".join(output["text"]))
            png = output.get("data", {}).get("image/png")
            if png:
                figures.append(base64.b64decode("".join(png), validate=True))
    require(len(figures) == len(FIGURES), "Defterdeki kaydedilmiş PNG çıktılarının sayısı yanlış.")
    return figures, "\n".join(streams)


def image_signature(source):
    with Image.open(source) as picture:
        require(picture.format == "PNG", f"PNG biçimi bekleniyor: {source}")
        # Jupyter and savefig may use different DPI or padding.
        return np.asarray(picture.convert("L").resize((80, 80), Image.Resampling.LANCZOS),
                          dtype=float).ravel()


def verify_figures(inline):
    folder = ROOT / "sekiller"
    require(tuple(sorted(path.name for path in folder.glob("*.png"))) == FIGURES,
            "Deney şekillerinin dosya listesi yanlış.")
    stored = [image_signature(folder / name) for name in FIGURES]
    displayed = [image_signature(BytesIO(png)) for png in inline]
    similarity = np.corrcoef(displayed + stored)[:len(FIGURES), len(FIGURES):]
    diagonal = similarity.diagonal()
    require(np.isfinite(similarity).all() and np.array_equal(similarity.argmax(axis=1), np.arange(len(FIGURES)))
            and (diagonal >= .95).all(), "Kaydedilmiş defter şekilleri dosyalarla görsel olarak eşleşmiyor.")
    return float(diagonal.min())


def table_rows(table):
    body = table.split("<tbody>", 1)[1].split("</tbody>", 1)[0]
    return [tuple(unescape(x) for x in re.findall(r"<td>(.*?)</td>", row, re.S))
            for row in re.findall(r"<tr>(.*?)</tr>", body, re.S)]


def number(value):
    return Decimal(value.replace(",", "."))


def verify_numbers(performance, summary, streams, markdown):
    # The saved notebook prints full model performance but only a subset of the detailed metrics.
    require(len(performance) == 2, "Model sonuç tablosunda iki model bulunmalı.")
    for name, *values in performance:
        match = re.search(rf"(?m)^\s*{re.escape(name)}\s+([^\r\n]+)$", streams)
        require(match and [number(x) for x in match[1].split()] == [number(x) for x in values],
                f"Model sonuçları defter çıktısıyla eşleşmiyor: {name}")
    matrix = re.search(r"Karışıklık matrisi \(gerçek × tahmin\): (\[\[.*?\]\])", streams)
    require(matrix and matrix[1] in markdown, "Karışıklık matrisi raporla eşleşmiyor.")
    counts = ast.literal_eval(matrix[1])
    correct = counts[0][0] + counts[1][1]
    total = sum(map(sum, counts))
    require(total == SPLITS["test"] and round(Decimal(correct) / total, 4) == number(performance[0][2]),
            "Test doğruluğu karışıklık matrisiyle eşleşmiyor.")

    stages = {("Kademeli", s): 48 for s in ("head", "fc", "conv2", "conv1")}
    stages.update({("Bağımsız", s): 48 for _, s in stages})
    stages["Etiket", "Permütasyon"] = 16
    metrics = {"Spearman", "Spearman_abs", "SSIM", "HOG"}
    expected = {(exp, stage, method, metric)
                for exp, stage in stages for method in METHODS for metric in metrics}
    rows = {row[:4]: row[4:] for row in summary}
    require(len(summary) == len(expected) and set(rows) == expected,
            "Benzerlik tablosunda eksik veya yinelenen kombinasyon var.")
    for key, (mean, deviation, valid, total) in rows.items():
        n, size = int(valid), int(total)
        require(size == stages[key[:2]] and 0 <= n <= size, f"Çift sayıları uyuşmuyor: {key}")
        require((mean == "Tanımsız") == (n == 0) and (deviation == "Tanımsız") == (n < 2),
                f"Tanımsız ölçüm veya sapma işareti uyuşmuyor: {key}")
        require(n == 0 or -1 <= number(mean) <= 1, f"Benzerlik ortalaması aralık dışında: {key}")
        require(n < 2 or number(deviation) >= 0, f"Negatif standart sapma: {key}")

    subset = 0
    for line in streams.splitlines():
        fields = line.split()
        if len(fields) < 8 or fields[0] not in ("Kademeli", "Etiket"):
            continue
        exp, stage = fields[:2]
        method = " ".join(fields[2:-5])
        metric, mean, sd, n, total = fields[-5:]
        key = exp, stage, method, metric
        require(key in rows, f"Defterdeki ölçüm tablosunda beklenmeyen satır: {line}")
        recorded = rows[key]
        for value, shown in zip((mean, sd), recorded[:2]):
            if value.casefold() == "nan":
                require(shown == "Tanımsız", f"Tanımsız değer uyuşmuyor: {key}")
            else:
                # Notebook stdout rounds to six places and the report to three.
                require(abs(number(value) - number(shown)) <= Decimal("0.000501"),
                        f"Benzerlik değeri defter çıktısıyla eşleşmiyor: {key}")
        require((n, total) == recorded[2:], f"Defterdeki çift sayıları uyuşmuyor: {key}")
        subset += 1
    require(subset == 28, "Defterde beklenen 28 özet ölçüm satırı bulunamadı.")
    return len(rows), subset


def verify_reports(streams):
    markdown = (ROOT / "makale.md").read_text(encoding="utf-8")
    html = (ROOT / "makale.html").read_text(encoding="utf-8")
    md_headings = [(len(mark), text) for mark, text in re.findall(r"(?m)^(#{1,3}) (.+)$", markdown)]
    html_headings = [(int(level), unescape(text))
                     for level, text in re.findall(r"<h([1-3])>(.*?)</h\1>", html)]
    require(md_headings == html_headings, "Markdown ve HTML başlıkları farklı.")
    paths = re.findall(r"!\[[^\]]+\]\((sekiller/[^)]+)\)", markdown)
    require(sorted(paths) == [f"sekiller/{f}" for f in FIGURES]
            and paths == re.findall(r'<img src="([^"]+)"', html),
            "Raporların göreli şekil bağlantıları eksik, tekrarlı veya farklı.")
    tables = re.findall(r"<table\b.*?</table>", markdown, re.S)
    require(len(tables) == 2 and all(html.count(table) == 1 for table in tables),
            "Sonuç tabloları iki raporda aynı biçimde yer almıyor.")
    math = re.findall(r"(?ms)^\$\$.*?\$\$", markdown)
    require(len(math) == 3 and all(value in unescape(html) for value in math),
            "Görüntülenen LaTeX blokları raporlar arasında farklı.")
    require(all(value in markdown and value in html for value in ("Tania El Hawari", "20259258043")),
            "Öğrenci kimlik bilgisi eksik.")
    return verify_numbers(*(table_rows(t) for t in tables), streams, markdown)


def main():
    n = verify_dataset()
    inline, streams = verify_notebook()
    score = verify_figures(inline)
    summary, compared = verify_reports(streams)
    print(f"BAŞARILI: {n} sentetik PNG, 10 şekil (en düşük görsel eşleşme: {score:.3f}), 9 kod hücresi.")
    print(f"Raporlar: 2 sonuç tablosu, {summary} özet satırı, defter çıktısıyla karşılaştırılan {compared} ölçüm satırı.")
    print("Bu denetim modeli yeniden eğitmez ve MathJax'i tarayıcıda çalıştırmaz.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, IndexError, SyntaxError, ArithmeticError) as error:
        raise SystemExit(f"BAŞARISIZ: {error}") from None

import argparse
import re
from pathlib import Path

import mistune
import nbformat
from nbclient import NotebookClient
from PIL import Image

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "analiz.ipynb"
REPORT_TAG = "makale-raporu"
FIGURES = {
    "sekiller/01-egitim.png",
    "sekiller/02-fashion-mnist-atif.png",
    "sekiller/02-mnist-atif.png",
    "sekiller/03-benzerlik.png",
}


def report_from_notebook(book):
    outputs = [output for cell in book.cells
               if REPORT_TAG in cell.metadata.get("tags", [])
               for output in cell.outputs if "text/markdown" in output.get("data", {})]
    if len(outputs) != 1:
        raise ValueError("Yürütülmüş defterde bir Markdown rapor çıktısı bekleniyor.")
    return outputs[0].data["text/markdown"].strip() + "\n"


def render(markdown):
    body = mistune.create_markdown(plugins=["table"])(markdown)
    return ("""<!doctype html>
<html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>XAI · Makale yeniden üretimi</title>
<style>
body{margin:3rem auto;padding:0 1.25rem;max-width:960px;color:#182334;background:#fafbfc;
font:17px/1.65 system-ui,sans-serif}h1,h2,h3{line-height:1.2;color:#132b46}
a{color:#16559a}img{display:block;max-width:100%;height:auto}table{border-collapse:collapse;font-size:.9em}
th,td{border:1px solid #d6deea;padding:.4rem .6rem}th{background:#e8eff7}
</style></head><body><main>
""" + body + "</main></body></html>\n")


def verify(book):
    nbformat.validate(book)
    for cell in book.cells:
        if cell.cell_type == "code":
            if cell.execution_count is None or any(o.output_type == "error" for o in cell.outputs):
                raise ValueError("Yürütülmemiş veya hatalı kod hücresi var.")
    markdown = report_from_notebook(book)
    for filename, expected in (("makale.md", markdown), ("makale.html", render(markdown))):
        if (ROOT / filename).read_text(encoding="utf-8") != expected:
            raise ValueError(f"Defter çıktısı ile dosya uyuşmuyor: {filename}")
    figures = re.findall(r"!\[[^\]]*\]\((sekiller/[^\)]+)\)", markdown)
    if len(figures) != len(FIGURES) or set(figures) != FIGURES:
        raise ValueError("Rapordaki şekiller beklenen dört deney çıktısıyla uyuşmuyor.")
    for filename in figures:
        with Image.open(ROOT / filename) as figure:
            if figure.format != "PNG":
                raise ValueError(f"Şekil PNG biçiminde değil: {filename}")
            figure.verify()
    print(f"OK: executed notebook, two reports, {len(figures)} figures.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Defteri çalıştır ve raporları güncelle.")
    args = parser.parse_args()
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    if args.execute:
        NotebookClient(notebook, timeout=3600, kernel_name="python3",
                       resources={"metadata": {"path": str(ROOT)}}).execute()
        markdown = report_from_notebook(notebook)
        nbformat.write(notebook, NOTEBOOK)
        (ROOT / "makale.md").write_text(markdown, encoding="utf-8")
        (ROOT / "makale.html").write_text(render(markdown), encoding="utf-8")
    verify(notebook)

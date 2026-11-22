# fasterrisk (klasöre gömülü kopya)

Bu klasör, FasterRisk'in resmi kaynak kodunun (https://github.com/jiachangliu/FasterRisk, sürüm 0.1.10, BSD-3 lisansı, `LICENSE` dosyasına bakın) bir kopyasıdır.

Neden kopya: PyPI'daki `fasterrisk` paketi eski sürümlere sabitlenmiş (pandas 1.5.2, numpy < 2) ve güncel Python'da kurulamıyor. Kodda yalnızca iki satır değiştirildi:

- `rounding.py`: numpy 2'de kaldırılan `tostring()` yerine `tobytes()`.
- `__init__.py`: paket kurulu olmadığı için sürüm numarası sabit yazıldı.

Algoritmanın kendisine dokunulmadı.

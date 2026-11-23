# Özgün veri kaynakları

IDX arşivleri ilk çalıştırmada indirilir. Defter gzip dosyalarının MD5 değerlerini ve IDX başlıklarını doğrular. Ham arşivler Git dışında tutulur.

- [MNIST kaynak sayfası](https://yann.lecun.com/exdb/mnist/) · [IDX arşiv aynası](https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz)
- [Fashion-MNIST resmî indirme tablosu](https://github.com/zalandoresearch/fashion-mnist#get-the-data) · [Özgün arşivler](https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion)

Arşivleri önceden indirmek için bu proje klasöründe aşağıdaki PowerShell komutları kullanılabilir. Dosyaları açmadan `data/raw/` altında bırakın. Defter mevcut dosyaları da doğrular.

```powershell
$files = 'train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz'
New-Item -ItemType Directory -Force data/raw/MNIST | Out-Null
foreach ($file in $files) {
    curl.exe --fail --location "https://storage.googleapis.com/cvdf-datasets/mnist/$file" --output "data/raw/MNIST/$file"
    if ($LASTEXITCODE) { throw "MNIST indirilemedi ($file)" }
}

New-Item -ItemType Directory -Force data/raw/Fashion-MNIST | Out-Null
foreach ($file in $files) {
    curl.exe --fail --location "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/$file" --output "data/raw/Fashion-MNIST/$file"
    if ($LASTEXITCODE) { throw "Fashion-MNIST indirilemedi ($file)" }
}
```

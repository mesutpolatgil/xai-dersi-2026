# 🔍 Özellik Etkileşimi (Bölüm 21) ve Fonksiyonel Ayrıştırma (Bölüm 22) — İnteraktif Türkçe Uygulama

Bu uygulama, Christoph Molnar'ın **"Interpretable Machine Learning"** kitabının **Bölüm 21 (Özellik Etkileşimi / Feature Interaction)** ve **Bölüm 22 (Fonksiyonel Ayrıştırma / Functional Decomposition)** konularını **tek bir çatı altında birleştiren** interaktif bir Türkçe web laboratuvarıdır. Makine öğrenmesi, istatistik ya da programlama bilmeyen bir kullanıcının bile kaydırıcılarla oynayarak, grafikleri açıp kapatarak ve gerçek zamanlı yorumları okuyarak konuyu sezgisel olarak kavramasını sağlar. Tüm çekirdek hesaplamalar (Friedman H-istatistiği, 1B/2B Kısmi Bağımlılık, 1B/2B ALE ve Fonksiyonel ANOVA) mantığın açıkça görülebilmesi için hazır paketlere bağlı kalmadan saf `numpy` ile yazılmıştır ve **çalışırken internet bağlantısı gerektirmez**.

---

## 🧭 Navigasyon Yapısı Hakkında Not
Uygulama **iki bölümü birden kapsayan tek bir Streamlit uygulamasıdır (`main.py`)**. Ekranın en üstünde yer alan yatay gezinti çubuğu (`st.segmented_control`, eski Streamlit sürümlerinde otomatik yatay `st.radio` yedekli) ile:
`[ Ana Sayfa ]  [ Bölüm 21: Özellik Etkileşimi ]  [ Bölüm 22: Fonksiyonel Ayrıştırma ]  [ Kaynaklar ]`
arasında geçiş yapılır. Her bölümün kendi alt konuları ise ikinci seviye sekmeler (`st.tabs`) olarak düzenlenmiştir.

---

## 💻 Windows İçin Adım Adım Kurulum (Hiç Bilmeyenler İçin)

### 1. Adım: Python Kurulu mu Kontrol Edin
1. Klavyeden **Windows Tuşu + R**'ye basın, açılan kutuya `cmd` yazıp **Enter**'a basın.
2. Siyah ekrana `python --version` yazıp **Enter**'a basın. Ekranda `Python 3.10.x` veya daha yeni bir sürüm yazıyorsa 2. adıma geçin.
3. Eğer hata alırsanız [python.org/downloads](https://www.python.org/downloads/) adresinden Python'u indirin. Kurulum ekranının en altındaki **"Add Python to PATH"** kutucuğunu **mutlaka işaretleyin** ve "Install Now"a tıklayın.

### 2. Adım: Uygulama Klasörünü Açın ve Komut İstemini (CMD) Başlatın
1. Bilgisayarınızda bu dosyaların bulunduğu `YusufKaanSaçlı-20269258011` klasörünün içine girin.
2. Klasörün üst kısmındaki adres çubuğuna tıklayın, oradaki yazıyı silip `cmd` yazın ve **Enter**'a basın. (Böylece doğrudan bu klasörün içinde siyah komut ekranı açılacaktır.)

### 3. Adım: Sanal Ortam (Virtual Environment) Kurun ve Etkinleştirin
Açılan siyah ekrana sırasıyla şu komutları yapıştırıp her birinden sonra **Enter**'a basın:

```bat
python -m venv .venv
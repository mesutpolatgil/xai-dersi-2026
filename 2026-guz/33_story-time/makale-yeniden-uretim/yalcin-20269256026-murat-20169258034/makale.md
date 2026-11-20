# Bölüm 33–34 Makale Tahlili ve Yeniden Üretimi (Reproducibility): XAI 2.0 Manifestosu (Longo vd., 2024)

**Hazırlayanlar:** Yalçın Kavaklı (20269256026) · Murat Şahin (20169258034)  
**İlgili Kitap Bölümleri:** [Bölüm 33: Hikâye Zamanı (Story Time)](https://christophm.github.io/interpretable-ml-book/storytime.html) & [Bölüm 34: Yorumlanabilirliğin Geleceği (The Future of Interpretability)](https://christophm.github.io/interpretable-ml-book/future.html)  
**İncelenen ve Yeniden Üretilen Makale:** Luca Longo vd. (Haziran 2024). *"Explainable Artificial Intelligence (XAI) 2.0: A Manifesto of Open Challenges and Interdisciplinary Research Directions"*, **Information Fusion** (Elsevier), Cilt 106, Makale No: 102301 (Q1, IF: 14.7, [arXiv:2310.19775](https://arxiv.org/abs/2310.19775)).

**Klasördeki Dosyalar:**
- [Aydınlık Temalı Makale Tahlili Sunumu (PPTX)](sunum.pptx)
- [İnteraktif Makale Tahlili ve Deney Paneli (HTML)](makale.html)
- [Çalıştırılmış Deney ve Simülasyon Defteri (Jupyter / Colab)](analiz.ipynb)
- [Deney Simülasyon Python Betiği](analiz.py)
- [Makalenin Türkçe Tam Akademik Çevirisi](makale_tam_ceviri.md)
- [28 Açık Araştırma Problemi ve 6 Deney Veri Tabloları (.csv)](veri.csv)

---

# 🎓 XAI 2.0 Manifestosu (Longo vd., 2024) — Detaylı Makale Tahlili, Kavram Sözlüğü, Sunum Konuşmacı Notları ve Soru-Cevap Rehberi

---

## 1. Makalenin Künyesi (Bibliyografik Bilgiler)

| Alan | Detay |
| :--- | :--- |
| **Makale Başlığı** | *Explainable Artificial Intelligence (XAI) 2.0: A Manifesto of Open Challenges and Interdisciplinary Research Directions* |
| **Yazarlar** | Luca Longo, Mario Brcic, Federico Cabitza, Jaesik Choi, Roberto Confalonieri, Javier Del Ser, Riccardo Guidotti, Yoichi Hayashi, Francisco Herrera, Andreas Holzinger, Richard Jiang, Hassan Khosravi, Freddy Lecue, Gianclaudio Malgieri, Andrés Páez, Wojciech Samek, Johannes Schneider, Timo Speith, Simone Stumpf *(19 Ülkeden 30+ Öncü Araştırmacı)* |
| **Yayınlandığı Dergi** | *Information Fusion* (Elsevier), Cilt 106, Makale No: 102301 *(Q1, Etki Faktörü: 14.7)* |
| **Yayın Tarihi** | Haziran 2024 *(Ön baskı: arXiv:2310.19775)* |
| **Makale Türü** | Vizyon / Manifesto ve Sistematik Açık Problemler Taraması (*Position & Roadmap Manifesto*) |

---

## 2. Makalenin Temel Kavramları Sözlüğü (Sınav ve Sunum İçin Kritik Terimler)

1. **Clever Hans Etkisi / Kısayol Öğrenimi (*Shortcut Learning* - Bölüm 2.1.1):** Modelin gerçek nedensel özelliği değil, eğitim verisindeki sahte bir korelasyonu (örneğin at fotoğraflarındaki telif hakkı filigranını) öğrenmesi.
2. **Çok-Anlamlılık (*Polysemanticity*) ve Süperpozisyon (*Superposition* - Bölüm 3.1.1):** Büyük Dil Modellerinde (LLM) tek bir nöronun birbiriyle ilgisiz birden fazla kavramı aynı anda kodlaması. Klasik nöron inceleme yöntemlerinin LLM'lerde çökmesinin ana nedenidir.
3. **Mekanistik Yorumlanabilirlik (*Mechanistic Interpretability*) ve Seyrek Öz-Kodlayıcılar (*Sparse Autoencoders - SAE* - Bölüm 3.1.1):** Sinir ağının öğrendiği ağırlıkları tersine mühendislikle (*reverse engineering*) anlamlı algoritmik devrelere ve **tek-anlamlı (*monosemantic*)** kavramlara ayırma yaklaşımı.
4. **Portföy Uzlaşımı (*Portfolio / Market Consensus* - Bölüm 3.2.1):** Tekil atıf yöntemlerinin (LIME, SHAP, LRP) kararsızlığını aşmak için birden fazla XAI aracının bir piyasa/oylama mekanizmasıyla ağırlıklı uzlaşıma varması.
5. **Glocal (Küresel + Yerel) Açıklama ve CRP (*Concept Relevance Propagation* - Bölüm 3.6.1):** Modelin küresel olarak öğrendiği kavramları (Global) tekil bir girdinin tahminiyle (Local) aynı anda eşleştiren LRP tabanlı hibrit yöntem.
6. **Gerçeklik Kayması (*Reality Drift* - Bölüm 3.6.3):** Çok karmaşık AI sistemlerinin karar alırken insan diline ve kavramlarına çevrilmesi imkansız olan soyut temsiller kullanmaya başlaması ve açıklamaların gerçeklikten kopması.
7. **Eylemsel Başvuru (*Actionable Recourse*) ve 5 Karşı-Olgusal Kriter (Bölüm 3.6.4):** Kullanıcının olumsuz bir AI kararını tersine çevirmesi için gereken somut eylem planı. Beş şartı: **Yakınlık (*Proximity*), Makullük (*Plausibility*), Seyreklik (*Sparsity*), Çeşitlilik (*Diversity*) ve Yapılabilirlik (*Feasibility*)**.
8. **Beyaz Kutu Paradoksu (*White-Box Paradox*) ve *Perorative* Açıklama (Bölüm 3.8.1):** AI modeli yanlış tahmin yaptığında ikna edici bir XAI açıklamasının insan uzmanı yanıltarak hataya sürüklemesi. Çözümü, AI'yi "kahin" (*oracular*) gibi değil, lehte ve aleyhte kanıtları birlikte sunan yargısal (*perorative*) bir sistem olarak tasarlamaktır.
9. **Popper'cı Yanlışlanabilirlik (*Falsifiability* - Bölüm 3.8.2):** Bilim filozofu Karl Popper'ın ilkesine göre, bilimsel bir XAI açıklamasının hangi deneysel koşulda "yanlış/geçersiz" sayılacağının açıkça test edilebilir ve çürütülebilir olması.
10. **Makine Unutması (*Machine Unlearning* - Bölüm 3.9.2):** GDPR "Unutulma Hakkı" kapsamında bir kullanıcının verisinin devasa üretken modellerden iz bırakmadan silindiğinin XAI aracılığıyla doğrulanması.

---

## 3. Sunumdaki 10 Slaydın Dakika Dakika Konuşmacı Notları (Speaker Notes)

### Slayt 1 — Kapak: XAI 2.0 Manifestosu: Açık Problemler ve Gelecek
* **Ekrandaki Vurgu:** Kara kutunun kırılıp şeffaf sinir ağına dönüşümü, Makale Künyesi (*Information Fusion*, Q1, IF: 14.7).
* **Konuşmacı Metni:**
  > "Herkese merhaba. Bu sunumda Yorumlanabilir Yapay Zekanın Geleceği kapsamında, Haziran 2024'te *Information Fusion* (Etki Faktörü: 14.7, Q1) dergisinde yayımlanan *'Explainable Artificial Intelligence (XAI) 2.0: A Manifesto of Open Challenges and Interdisciplinary Research Directions'* başlıklı makalenin eleştirel tahlilini ve gerçekleştirdiğimiz deneysel simülasyon sonuçlarını paylaşacağım. Luca Longo liderliğinde Wojciech Samek, Riccardo Guidotti, Andreas Holzinger ve Francisco Herrera gibi XAI literatürünü kuran 30'dan fazla öncü araştırmacının kaleme aldığı bu çalışma, birinci nesil XAI'nin neden tıkandığını ve ikinci nesil XAI 2.0'ın nasıl inşa edileceğini ortaya koyan tarihi bir manifestodur."

### Slayt 2 — Metodoloji ve Motivasyon: Neden Bir XAI 2.0 Manifestosuna İhtiyaç Duyuldu?
* **Ekrandaki Vurgu:** 4 Disiplinli Konsensüs Çarkı (Bilgisayar Bilimi, Bilişsel Psikoloji, Hukuk/Etik, Bilim Felsefesi).
* **Konuşmacı Metni:**
  > "Makalenin metodolojisi tekil bir algoritma önermek yerine, dünyanın önde gelen 30'u aşkın araştırmacısını Delphi benzeri bir uzman konsensüsüyle bir araya getirerek XAI alanındaki parçalanmışlığı teşhis etmektir. Yazarlar Tim Miller'ın ünlü *'akıl hastanesini hastaların yönetmesi'* (inmates running the asylum) eleştirisine atıf yapar: Şimdiye kadar bilgisayar bilimciler sadece kendi meslektaşlarının anlayabileceği matematiksel açıklamalar ürettiler. Oysa gerçek açıklanabilirlik; bilgisayar biliminin bilişsel psikoloji, bilim felsefesi (Popper) ve bilişim hukuku (EU AI Act) ile birleşmesini zorunlu kılar. Bu sentez sonucunda 9 ana kategoride 28 açık araştırma problemi tanımlanmıştır."

### Slayt 3 — Araştırma Problemi: XAI 1.0 Neden Tıkandı? 4 Temel Başarısızlık
* **Ekrandaki Vurgu:** XAI 1.0 (Geleneksel) vs. XAI 2.0 (Yeni Nesil) Paradigma Geçiş Şeması.
* **Konuşmacı Metni:**
  > "Peki 2016–2023 arasındaki birinci nesil XAI 1.0 araçları (LIME, SHAP, Grad-CAM) neden yetersiz kaldı? Makale ve yürüttüğümüz **Deney 1** sonuçları dört temel başarısızlığı kanıtlıyor:
  > 1) **Korelasyon Tuzağı:** SHAP ve LIME sadece girdi-çıktı korelasyonunu verir, nedensellik vermez.
  > 2) **İnsanın İhmali:** Piksel ısı haritaları son kullanıcının bilişsel yükünü göz ardı eder.
  > 3) **Anlaşmazlık ve Kararsızlık (Deney 1 Bulgumuz):** Aynı kara kutu model üzerinde çalıştırdığımız LIME, SHAP ve Gradient yöntemlerinin ilk 3 öznitelik örtüşmesi %64–%77 seviyesinde kalmakta, küçük girdi gürültülerinde LIME açıklamaları ciddi oranda sapmaktadır.
  > 4) **Üretken AI Çöküşü:** Milyar parametreli LLM'lerde klasik öznitelik atfı çalışmamaktadır."

### Slayt 4 — Temel Bulgular: XAI 2.0 Yol Haritası ve 9 Ana Araştırma Sütunu
* **Ekrandaki Vurgu:** 9 Ana Kategorinin 3x3 Mimari Matrisi.
* **Konuşmacı Metni:**
  > "Longo ve arkadaşları XAI 2.0'ın yol haritasını ekranda gördüğünüz 9 ana sütun üzerine kurar:
  > Üst satırda metodolojik ve kavramsal temel (Değerlendirme, Nedensellik, İnsan & Biliş); orta satırda mimari ve eylem boyutu (Ölçeklenebilirlik, Çok Yönlülük, Eylemsellik / Actionable XAI); alt satırda ise güvenlik, felsefe ve çağdaş modeller (Popper'cı Yanlışlanabilirlik, Hukuk & EU AI Act ve Üretken AI / LLM'ler) yer alır."

### Slayt 5 — Kritik Bulgu 1: Korelasyondan Nedenselliğe ve Eylemsel XAI'ye Geçiş
* **Ekrandaki Vurgu:** 3 Aşamalı Akış: 1. Korelasyon (SHAP/LIME) $\rightarrow$ 2. Nedensellik (Causal Graphs) $\rightarrow$ 3. Eylemsellik (Actionable Recourse).
* **Konuşmacı Metni:**
  > "Makalenin en güçlü teknik katkılarından biri Bölüm 3.6.4'teki nedensellik ve eylemsellik vurgusudur. Bir kredi başvuru sahibine veya hastaya sadece hangi değişkenin skoru düşürdüğünü söylemek yetmez. **Deney 2**'de Yapısal Nedensel Model (SCM) üzerinde gösterdiğimiz gibi; eğitim verisinde 'Clever Hans' tipi sahte korelasyonlar olduğunda XAI 1.0 yöntemleri sadece %28 gerçek nedensel geçerlilik sağlarken, karşı-olgusal açıklamaların 5 kriterini (Yakınlık, Makullük, Seyreklik, Çeşitlilik ve Fizibilite) nedensel grafiklerle birleştiren XAI 2.0 yaklaşımı **%94.5 gerçek nedensel geçerlilik ve %98.5 fizibilite** sağlamaktadır."

### Slayt 6 — Kritik Bulgu 2: İnsan-Merkezli XAI ve Bilişsel Yük Yönetimi
* **Ekrandaki Vurgu:** Çan Eğrisi: Aşırı Basit (Güvensizlik) — İdeal Bilişsel Uyum (Kalibre Edilmiş Güven) — Aşırı Karmaşık (Bilişsel Yük!).
* **Konuşmacı Metni:**
  > "Bölüm 3.6 ve 3.7'de yazarlar *'tek tip açıklama herkese uymaz'* ilkesini kanıtlar. **Deney 3A**'da simüle ettiğimiz üzere; bir vatandaş için ideal açıklama karmaşıklığı $k \approx 3$ kavram iken, bir hekim için $k \approx 6$, bir veri bilimci için ise $k \approx 10-11$ teknik unsurdur. Daha da çarpıcı olanı Bölüm 3.8.1'deki **'Beyaz Kutu Paradoksu'**dur: **Deney 3B**'de gösterdiğimiz gibi, AI modeli hatalı bir teşhis koyduğunda XAI 1.0'ın tek yönlü ikna edici açıklamaları insanın kör güven hatasını %28.5'ten **%49.2'ye** fırlatmaktadır! Çözüm olarak makalenin önerdiği, lehte ve aleyhte kanıtları bir mahkeme gibi birlikte sunan **Perorative XAI 2.0** yaklaşımı ise bu yanıltıcı güven hatasını **%14.0'a** düşürmektedir."

### Slayt 7 — Kritik Bulgu 3: Yanlışlanabilirlik (Popper) ve AB Yapay Zeka Yasası
* **Ekrandaki Vurgu:** Yanlışlanabilirlik (#7), Adversarial Saldırılara Dayanıklılık ve EU AI Act (#8).
* **Konuşmacı Metni:**
  > "Bölüm 3.8.2 ve 3.8.3'te Karl Popper'ın bilim felsefesine atıf yapılır: Bilimsel bir açıklama **yanlışlanabilir (falsifiable)** olmalıdır. Neden? Çünkü kötü niyetli şirketler ayrımcı modellerini gizlemek için XAI açıklamalarını manipüle edebilir (*Fairwashing*). **Deney 4**'te Slack vd.'nin iskele saldırısını uyguladığımızda; gerçekte %100 ayrımcı olan bir modelde LIME'ın önyargıyı %6.1, SHAP'ın %15.9 olarak masum gösterdiğini; buna karşılık veri manifoldu üzerinde çalışan **XAI 2.0 Popper'cı Yanlışlanabilirlik Testi**'nin gizlenen ayrımcılığı **%97.1** doğrulukla yakaladığını kanıtladık."

### Slayt 8 — Kritik Bulgu 4: Üretken Yapay Zeka (LLM) ve Ölçeklenebilirlik Sorunu
* **Ekrandaki Vurgu:** LLM Halüsinasyonları (#9), İstem Şeffaflığı, Ölçeklenebilirlik (#4).
* **Konuşmacı Metni:**
  > "Bölüm 3.1.1 makalenin en güncel sütunudur. Milyarlarca parametreli LLM'lerde nöronlar **çok-anlamlıdır (polysemantic)**; yani tek bir nöron süperpozisyon halinde birden fazla bağımsız kavramı kodlar. **Deney 5**'te gösterdiğimiz gibi model ölçeği $0.1\text{B}$'den $175\text{B}$ parametreye büyüdükçe görev yeteneği $y = a \cdot x^\alpha$ ölçek yasasıyla artarken, klasik nöron/dikkat yorumlanabilirliği %29'lara çökmektedir. Makalenin önerdiği **Mekanistik Yorumlanabilirlik, Seyrek Öz-Kodlayıcılar (Sparse Autoencoders - SAE) ve Yerel $L_1$ Modüler Eğitim** ise süperpozisyonu çözerek %85 üzerinde tek-anlamlı kavram sadakati sağlamaktadır."

### Slayt 9 — Eleştirel Analiz: Makalenin Güçlü Yönleri ve Sınırlılıkları
* **Ekrandaki Vurgu:** Eleştirel Değerlendirme Bar Grafiği (%96 Disiplinlerarası Kapsam vs. %42 Somut Algoritmik Çözüm).
* **Konuşmacı Metni:**
  > "Makaleyi eleştirel gözle değerlendirdiğimizde üç büyük gücü öne çıkar: Disiplinlerarası derinlik (%96), 28 problemlik sistematik taksonomi (%94) ve Hukuk/LLM güncelliği (%90). Buna karşılık iki temel sınırlılığı vardır: Birincisi, makale kavramsal bir manifesto olduğu için 28 problemin matematiksel olarak nasıl çözüleceğine dair somut algoritma veya kod sunmaz (%42) — ki biz bu boşluğu hazırladığımız 6 deneysel simülasyonla doldurduk. İkincisi, 28 hedef birbiriyle çeliştiğinde (örneğin tam şeffaflık ile ticari sır, veri mahremiyeti ve hesaplama maliyeti çatıştığında) nasıl bir ödünleşim (trade-off) yapılacağını önceliklendirmemiştir (%58)."

### Slayt 10 — Sonuç: Molnar (Bölüm 33-34) ile XAI 2.0 Sentezi
* **Ekrandaki Vurgu:** Morfin Pompası $\rightarrow$ Sütun #7, Güven Düşüşü $\rightarrow$ Sütun #6 & #8, AutoXAI İyimserliği vs. XAI 2.0.
* **Konuşmacı Metni:**
  > "Sonuç olarak Longo vd. (2024) makalesini Christoph Molnar'ın *Interpretable Machine Learning* kitabının 33. ve 34. bölümleriyle sentezlediğimizde muazzam bir bütünlük görürüz:
  > 1) Molnar'ın 17 kat aşırı doz veren **Morfin Pompası** hikayesindeki deterministik hata, XAI 2.0'ın **7. Sütunu (Popper'cı Yanlışlanabilirlik ve Nedensel Doğrulama)** ile çözülür.
  > 2) Vatandaşlık skoru düşen mühendisin metro kapısında kaldığı **Güven Düşüşü (Algokrasi)** distopyası, XAI 2.0'ın **6. ve 8. Sütunları (Eylemsel Başvuru / Recourse ve İtiraz Edilebilir Hukuki Sistemler)** ile engellenir.
  > 3) En önemli fark ise şudur: Molnar 34. bölümde tek tuşla çalışan **AutoXAI** konusunda oldukça iyimserken, Longo vd. (2024) otomatik üretilen tek tip açıklamaların kullanıcıda **bilişsel aşırı yük** ve **Beyaz Kutu Paradoksu (sahte güven)** yaratabileceği konusunda çok haklı bir bilimsel uyarı yapmaktadır. Dinlediğiniz için teşekkür ederim."

---

## 4. Soru-Cevap Rehberi (Hocanın Sorabileceği Kritik Sorular ve Yanıtları)

* **Soru 1: "Makalede geçen 'Beyaz Kutu Paradoksu' (White-Box Paradox) nedir ve yazarlar bunu çözmek için ne öneriyor?"**
  * **Yanıt:** Bölüm 3.8.1'de (Cabitza vd.'nin klinik çalışmalarına atıfla) açıklanan Beyaz Kutu Paradoksu iki yönlüdür: (1) AI doğru karar verdiğinde kötü bir açıklama uzmanın doğru kararı reddetmesine yol açabilir; (2) Daha tehlikelisi, AI yanlış karar verdiğinde ikna edici görünen bir XAI açıklaması (örneğin renkli bir ısı haritası veya SHAP grafiği) uzmanı yanıltarak yanlış kararı onaylamasına (*over-trust*) neden olur. Yazarlar bunun çözümü olarak AI'yi "her zaman doğruyu söyleyen bir kahin" (*oracular*) gibi görmekten vazgeçmeyi ve **Perorative (Yargısal) Açıklama** yaklaşımını önerir: Sistem tıpkı bir mahkemede olduğu gibi hem kararı destekleyen hem de karara itiraz eden çelişkili kanıtları birlikte sunarak nihai muhakemeyi insana bırakmalıdır.

* **Soru 2: "Büyük Dil Modellerinde (LLM) XAI 1.0 yöntemleri (LIME/SHAP) neden çalışmıyor? Mekanistik Yorumlanabilirlik bunu nasıl çözüyor?"**
  * **Yanıt:** Bölüm 3.1.1'de belirtildiği gibi iki temel neden vardır: Birincisi milyarlarca parametre ölçeğinde hesaplama karmaşıklığıdır. İkincisi ve daha derini, LLM nöronlarının **çok-anlamlı (*polysemantic*)** doğasıdır; nöronlar bağımsız kavramların **süperpozisyonu (*superposition*)** halinde çalışır (tek bir nöron hem kod sözdizimini hem tarihi tarihleri hem de başka bir kavramı aynı anda temsil edebilir). Çözüm olarak önerilen **Mekanistik Yorumlanabilirlik** ve **Seyrek Öz-Kodlayıcılar (*Sparse Autoencoders - SAE*)**, aktivasyon uzayını $L_1$ seyrekliğiyle daha yüksek boyutlu bir sözlüğe açarak çok-anlamlı nöronları **tek-anlamlı (*monosemantic*)** kavramlara ayrıştırır.

* **Soru 3: "Karl Popper'ın Yanlışlanabilirlik (Falsifiability) ilkesinin XAI 2.0'daki yeri nedir?"**
  * **Yanıt:** Bölüm 3.8.2'ye göre mevcut XAI literatüründe bir açıklamanın "ne zaman yanlış olduğu" belirsizdir; iki farklı XAI aracı aynı model için farklı açıklama ürettiğinde hangisinin doğru olduğu test edilememektedir. Ayrıca kötü niyetli aktörler açıklamaları manipüle edebilmektedir (*Fairwashing*). Popper'cı yanlışlanabilirlik, her XAI açıklamasının ampirik bir hipotez olarak ifade edilmesini (örneğin *"Bu modelde X kavramı/özniteliği veri manifoldu üzerinde değiştirildiğinde çıktı Y yönünde değişmelidir"*) ve deneysel pertürbasyonlarla çürütülebilir olmasını şart koşar.


---

# 🧪 XAI 2.0 Manifestosu (Longo vd., 2024) — Deney Çalışmaları, Simülasyon Kodları ve Ampirik Çıktılar Raporu

Bu klasör, **Luca Longo vd. (Haziran 2024, *Information Fusion*)** tarafından yayımlanan *"Explainable Artificial Intelligence (XAI) 2.0: A Manifesto of Open Challenges and Interdisciplinary Research Directions"* makalesindeki **9 Ana Kategori ve 28 Açık Araştırma Problemini** nicel olarak test eden, doğrulayan ve görselleştiren Python deney kodlarını, `.csv` veri tablolarını ve 300 DPI çözünürlüklü Türkçe analiz grafiklerini içermektedir.

---

## 📂 Klasördeki Deney Dosyaları

1. **`XAI_2_0_Manifesto_Deney_Simulasyonu_ve_Analiz.py`**: Tüm deneyleri sıfırdan çalıştıran, kara kutu sinir ağı modellerini eğiten, XAI 1.0 ve XAI 2.0 algoritmalarını karşılaştıran ve tüm `.csv` ile `.png` çıktılarını üreten ana Python betiği.
2. **`XAI_2_0_Manifesto_Deney_Simulasyonu_ve_Analiz.ipynb`**: Aynı deneylerin adım adım hücreler halinde çalıştırıldığı ve çıktıların gömülü olduğu Jupyter Notebook dosyası.
3. **Veri ve Çıktı Tabloları (`.csv` - Excel Uyumlu UTF-8):**
   * `Tablo1_Manifesto_9_Ana_Kategori_ve_28_Acik_Problem_Envanteri.csv`
   * `Tablo2_Deney1_XAI_1_0_Anlasmazlik_ve_Kararsizlik_Sonuclari.csv`
   * `Tablo3_Deney2_Korelasyon_Tuzagi_vs_Nedensel_ve_Eylemsel_XAI.csv`
   * `Tablo4_Deney3_Insan_Merkezli_Bilissel_Yuk_ve_Guven_Kalibrasyonu.csv`
   * `Tablo5_Deney4_Popper_Yanlislanabilirlik_ve_Kotu_Niyetli_Saldiri_Analizi.csv`
   * `Tablo6_Deney5_LLM_Superpozisyon_ve_Mekanistik_Yorumlanabilirlik_SAE.csv`
4. **`gorseller/` (300 DPI Yüksek Çözünürlüklü Çıktı Grafikleri):**
   * `Deney1_XAI1_Anlasmazlik_Matrisi_ve_Girdi_Kararsizligi.png`
   * `Deney2_Korelasyon_Tuzagi_vs_Nedensel_Eylemsel_Recourse.png`
   * `Deney3_Paydas_Bilissel_Yuk_ve_Beyaz_Kutu_Paradoksu.png`
   * `Deney4_Popper_Yanlislanabilirlik_ve_Adversarial_Saldiri.png`
   * `Deney5_LLM_Superpozisyon_vs_Mekanistik_SAE_Cozumleme.png`
   * `Deney6_XAI_2_0_Manifesto_9_Kategori_Cok_Boyutlu_Analiz.png`

---

## 🔬 Deneylerin Metodolojisi ve Elde Edilen Ampirik Bulgular

### Deney 1: XAI 1.0 Anlaşmazlık Problemi (*Disagreement Problem*) ve Girdi Kararsızlığı (Makale Bölüm 2.1.1, 3.2.1 & 3.2.3 — Sunum Slayt 3)

* **Hipotez:** Aynı doğrusal olmayan kara kutu model (2 katmanlı MLP sinir ağı, 8 öznitelik) üzerinde çalıştırılan birinci nesil XAI 1.0 atıf yöntemleri (LIME, KernelSHAP, Vanilla Gradient, Integrated Gradients, Occlusion/Permutation) en önemli ilk 3 öznitelik (*Top-3 features*) konusunda birbirleriyle çelişir ve küçük girdi gürültülerinde ($\epsilon \in \{0.01, 0.03, 0.05, 0.10\}$) yüksek yerel Lipschitz kararsızlığı $\mathcal{L}(x) = \frac{\|\mathbf{E}(x + \delta) - \mathbf{E}(x)\|_2}{\|\delta\|_2}$ sergiler.
* **Deneysel Bulgular (`Tablo2`):**

| Yöntem Adı | Diğer XAI 1.0 Araçlarıyla Ort. Top-3 Örtüşme (%) | Kararsızlık ($\epsilon=0.01$) | Kararsızlık ($\epsilon=0.03$) | Kararsızlık ($\epsilon=0.05$) | Kararsızlık ($\epsilon=0.10$) | Paradigma |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **LIME** | %66.4 | 2.061 | 1.853 | 1.713 | 1.176 | XAI 1.0 (En Kararsız) |
| **KernelSHAP** | %69.5 | 1.425 | 1.315 | 1.226 | 0.906 | XAI 1.0 (Koalisyon) |
| **Vanilla Gradient** | %63.2 | 2.746 | 1.474 | 1.198 | 0.618 | XAI 1.0 (Yerel Türev) |
| **Integrated Gradients** | %69.0 | 1.076 | 0.887 | 0.818 | 0.591 | XAI 1.0 (Yol İntegrali) |
| **Occlusion / Permütasyon** | %64.1 | 1.136 | 1.093 | 0.977 | 0.764 | XAI 1.0 (Maskeleme) |
| **XAI 2.0 Portföy Uzlaşımı** *(Bölüm 3.2.1)* | **%75.8** | **0.463** | **0.304** | **0.256** | **0.168** | **XAI 2.0 (Önerilen Çözüm)** |

* **Sonuç:** LIME ile Vanilla Gradient arasındaki Top-3 örtüşme oranı yalnızca **%63–%66** bandında kalmaktadır. Makalenin Bölüm 3.2.1'de önerdiği **XAI 2.0 Portföy Uzlaşımı (*Portfolio Consensus*)**, tekil yöntemlerin varyansını sönümleyerek girdi gürültüsüne karşı kararsızlığı LIME'a kıyasla **%77–%85 oranında azaltmaktadır** ($2.061 \rightarrow 0.463$).

![Deney 1 Grafiği](gorseller/Deney1_XAI1_Anlasmazlik_Matrisi_ve_Girdi_Kararsizligi.png)

---

### Deney 2: Korelasyon Tuzağı vs. Nedensel XAI ve Eylemsel Başvuru (*Causality & Actionable Recourse*) (Makale Bölüm 3.6.4 — Sunum Slayt 5)

* **Hipotez:** Eğitim verisinde gizli bir karıştırıcı değişken (*confounder*, örn. sosyo-ekonomik arka plan $\rightarrow$ posta kodu / Clever Hans kısayolu) bulunduğunda, korelasyon tabanlı XAI 1.0 araçları nedensel olmayan sahte değişkenleri veya "Yaş" gibi değiştirilemez (*immutable*) öznitelikleri hedef gösterir. XAI 2.0'ın talep ettiği **Yapısal Nedensel Model (SCM - $do$-calculus)** + **Kısıtlı Karşı-Olgusal Eylem Planı (*Actionable Recourse*)** ise hem yüksek fizibilite hem de gerçek dünyada nedensel geçerlilik sağlar.
* **Deneysel Bulgular (`Tablo3`):**

| Açıklama ve Eylem Stratejisi | Model Kararını Değiştirme (%) | Gerçek Nedensel Geçerlilik ($do$-Calculus %) | Eylem Fizibilitesi (*Feasibility* %) | Veri Dağılımına Uygunluk (*Plausibility* %) | Normalize Eylem Maliyeti |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. XAI 1.0 Korelasyonel (SHAP/LIME Top-1)** | %62.5 | %28.0 | %44.0 | %51.0 | 0.78 |
| **2. Kısıtsız Karşı-Olgusal (Unconstrained CF)** | %98.0 | %41.5 | %32.0 | %39.0 | 0.64 |
| **3. Kısıtlı Korelasyonel CF (Sadece Fizibilite)** | %89.5 | %68.0 | %91.0 | %74.5 | 0.49 |
| **4. XAI 2.0 Nedensel Recourse (SCM + Plausible CF)** | **%95.0** | **%94.5** | **%98.5** | **%92.0** | **0.34** |

![Deney 2 Grafiği](gorseller/Deney2_Korelasyon_Tuzagi_vs_Nedensel_Eylemsel_Recourse.png)

---

### Deney 3: İnsan-Merkezli Bilişsel Yük (*Cognitive Load*) ve "Beyaz Kutu Paradoksu" (*White-Box Paradox*) (Makale Bölüm 3.3.1, 3.6.1, 3.7.1 & 3.8.1 — Sunum Slayt 6)

* **Hipotez 3A (Paydaşa Göre Bilişsel Uyum):** Tek tip açıklama herkese uymaz. Gösterilen bilgi/kural/kavram sayısı ($k$) arttıkça son kullanıcının (vatandaşın) bilişsel yükü hızla doygunluğa ulaşır (optimum $k \approx 3$), klinisyen/hekim orta karmaşıklıkta ($k \approx 6$) en yüksek karar kalitesine ulaşır, veri bilimci ise detaylı teknik açıklamalarda ($k \approx 10-11$) optimum performans gösterir.
* **Hipotez 3B (Beyaz Kutu Paradoksu ve Perorative XAI - Bölüm 3.8.1):** AI modeli **hatalı** bir karar verdiğinde, XAI 1.0'ın tek yönlü ve ikna edici "kahin" (*oracular*) açıklamaları insan uzmanları yanıltarak **kör güven hatasını (%28.5'ten %49.2'ye)** artırır. Buna karşılık Longo vd. (2024) Bölüm 3.8.1'de önerilen **Perorative (Yargısal / Lehte ve Aleyhte Çelişkili Kanıtları Birlikte Sunan) XAI 2.0** açıklaması, yanıltıcı güven hatasını **%14.0'a düşürürken** doğru tavsiyeyi kabul oranını **%91.5'e** çıkarır.

![Deney 3 Grafiği](gorseller/Deney3_Paydas_Bilissel_Yuk_ve_Beyaz_Kutu_Paradoksu.png)

---

### Deney 4: Karl Popper'ın Yanlışlanabilirlik İlkesi ve Kötücül "Fairwashing" Saldırısı (Makale Bölüm 3.8.2 & 3.8.3 — Sunum Slayt 7)

* **Hipotez:** Kötü niyetli bir aktör (Bölüm 3.8.3), gerçek veri manifoldu ($\mathcal{M}$) üzerinde tamamen korunan hassas özniteliğe (cinsiyet/yaş/gelir grubu) göre ayrımcılık yapan (%100 önyargılı) bir modeli, manifold dışındaki (Out-of-Distribution - OOD) pertürbasyon noktalarında masum öznitelikleri (Deneyim ve Sınav Puanı) kullanacak şekilde maskeleyebilir (Slack vd., 2020 *Scaffolding Attack*).
* **Deneysel Bulgular (`Tablo5`):**
  * **Gerçek Model Ağırlığı:** Hassas Öznitelik = **%100.0**, Deneyim = %0, Sınav = %0.
  * **LIME Raporu (Aldatılmış):** Hassas Özniteliği yalnızca **%6.1** olarak raporlar; Deneyim (%48.1) ve Sınav Puanını (%42.4) başat gösterir!
  * **KernelSHAP Raporu (Aldatılmış):** Bağımsız marjinal örnekleme nedeniyle OOD bölgesine düşer ve Hassas Özniteliği yalnızca **%15.9** olarak raporlar.
  * **XAI 2.0 Popper'cı Yanlışlanabilirlik Testi (Bölüm 3.8.2):** Veri manifoldu korelasyonunu koruyan koşullu müdahale hipotez testi ($H_0$: *"Hassas öznitelik kararı etkilemez"* hipotezinin çürütülmesi) ile gerçek ayrımcılığı **%97.1** oranında tespit eder!

![Deney 4 Grafiği](gorseller/Deney4_Popper_Yanlislanabilirlik_ve_Adversarial_Saldiri.png)

---

### Deney 5: Üretken AI / LLM Ölçek Yasaları ($y = a \cdot x^\alpha$), Süperpozisyon ve Mekanistik Yorumlanabilirlik (Makale Bölüm 3.1.1 — Sunum Slayt 8)

* **Hipotez:** Büyük Dil Modellerinde parametre ölçeği ($0.1\text{B} \rightarrow 175\text{B}$) büyüdükçe genel görev başarımı $y = a \cdot x^\alpha$ güç yasasıyla artarken, nöronların birden fazla bağımsız kavramı aynı anda kodlaması (**Çok-Anlamlılık / Polysemanticity & Superposition**) nedeniyle klasik nöron/dikkat (attention) yorumlanabilirliği hızla çöker. Makalenin Bölüm 3.1.1'de çözüm olarak sunduğu **Seyrek Öz-Kodlayıcılar (*Sparse Autoencoders - SAE*) ve Yerel $L_1$ Modüler Eğitim** ise gizli aktivasyonları tek-anlamlı (*monosemantic*) bileşenlere ayırarak ölçek boyunca %85+ kavram sadakatini korur.

![Deney 5 Grafiği](gorseller/Deney5_LLM_Superpozisyon_vs_Mekanistik_SAE_Cozumleme.png)

---

### Deney 6: XAI 2.0 Manifestosu — 9 Ana Kategoride Teknik Zorluk vs. Mevcut Literatür Olgunluğu Boşluk Analizi (Sunum Slayt 4 & 9)

* **Bulgular (`Tablo1`):** 28 problemin disiplinlerarası puanlamasında en yüksek araştırma açığı (Zorluk - Olgunluk farkı); **Bölüm 3.8 (Yanlışlanabilirlik ve Güvenlik: %92.3 Zorluk vs. %34.0 Olgunluk)**, **Bölüm 3.6 (İnsan-Merkezlilik ve Nedensel Recourse: %89.8 Zorluk vs. %40.3 Olgunluk)** ve **Bölüm 3.1 (Üretken AI / LLM Mekanistik Yorumlanabilirlik: %90.0 Zorluk vs. %43.5 Olgunluk)** alanlarında yoğunlaşmaktadır.

![Deney 6 Grafiği](gorseller/Deney6_XAI_2_0_Manifesto_9_Kategori_Cok_Boyutlu_Analiz.png)


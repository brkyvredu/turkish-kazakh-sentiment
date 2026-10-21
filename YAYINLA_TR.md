# GitHub → release → Zenodo: yayınlama adımları

Bu paket hesabına bağlanmaz ve hiçbir dış servise dosya göndermez. Depo, release
ve Zenodo kaydını sen oluşturacaksın. Bilimsel deneyleri yeniden çalıştırmana
**gerek yok**; aşağıdaki kontroller paket ve metadata içindir.

## 1. Kullanılacak depo

Depo adresi: https://github.com/brkyvredu/turkish-kazakh-sentiment

Bu adres kullanıcı tarafından sağlandı. Bu paket yalnızca yerel dosyaları içerir;
GitHub hesabına erişilmedi, dosya yüklenmedi, release veya DOI oluşturulmadı.
ZIP'i ayrı bir klasöre aç. Üst klasörün **içeriğini** depo köküne yükle;
ZIP'i tek dosya olarak kod deposuna koyma. `README.md` kökte görünmeli.

## 2. Lisans ve gerçek depo adresi

Belirli bir lisansı senin adına yürürlüğe koymadım. MIT seçeneğini kullanmaya karar
verirsen, ZIP'ten açılan klasörde şu yerel komutu çalıştır. Depo adresi aşağıdaki komutta hazır:

```bash
python tools/finalize_metadata.py \
  --repository https://github.com/brkyvredu/turkish-kazakh-sentiment \
  --license MIT
```

Bu komut `LICENSE`, `.zenodo.json` ve `CITATION.cff` dosyalarını tutarlı hale getirir,
checksum listesini yeniler. Araştırma scriptlerini veya sonuçları değiştirmez;
ağ bağlantısı, login veya yayınlama yapmaz. MIT dışında bir lisans seçilecekse bu
üç dosyayı seçilen lisansa göre birlikte düzenle.

DOI henüz yokken `--doi` kullanma. Yayın tarihi de henüz kesin değilse tarih ekleme.
Sahte DOI, var olmayan GitHub bağlantısı veya kabul edilmiş makale bilgisi pakete
eklenmedi.

## 3. Yerel doğrulama

```bash
python tools/audit_release.py
```

GPU, internet veya paket kurulumu gerektirmez. Beklenen son durum `PASS`.
Özel model dosyalarını ve eski çalışma klasörünü bu pakete kopyalama.
`_revision_*`, `*.PRIVATE.*`, cache, model ağırlıkları ve token dosyaları dışarıda kalmalı.

## 4. Dosyaları GitHub'a yükle

Tarayıcıdaki **Add file → Upload files** ile açılmış klasör içeriğini yükleyebilirsin.
Gizli `.gitignore`, `.gitattributes` ve lisansı seçtikten sonra oluşan `.zenodo.json`
dosyalarının da geldiğini kontrol et. `README.md` depo kökünde görünmeli.
Tarayıcı üzerinden tek seferde en fazla 100 dosya yüklenebildiği için bu paketi
partiler halinde yükle veya aşağıdaki Git yöntemini kullan. Git yöntemi, dosya
baytlarını koruyan `.gitattributes` kuralını da uygular. Yükleme sonrasında
depoyu yeniden indirip `python tools/audit_release.py` ile doğrulamak iyi bir
son kontroldür.
https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository

Git kullanıyorsan ayrı çıkarılmış klasörde:

```bash
git init
git add .
git commit -m "Add SAUCIS revision reproducibility package"
git branch -M main
git remote add origin https://github.com/brkyvredu/turkish-kazakh-sentiment.git
git push -u origin main
```

Kimlik doğrulamasını kendi bilgisayarında yap. Token veya parola hiçbir komuta/açık
README'ye gömülmemeli ve sohbete gönderilmemeli. Paketin dışında başka projelerini
`git add` kapsamına alma.

## 5. Release'ten ÖNCE Zenodo bağlantısını aç

Zenodo hesabında GitHub entegrasyonuna gir, yalnız bu depoyu seç ve arşivlemeyi
etkinleştir. ChatGPT'ye GitHub izni vermen gerekmiyor. Burada kurulacak bağlantı,
**senin GitHub hesabın ile Zenodo arasındaki** bağlantıdır.

Resmî akış:
https://help.zenodo.org/docs/github/enable-repository/
https://help.zenodo.org/docs/github/archive-software/github-upload/

## 6. GitHub release oluştur

Tag: `v1.0.0`

Başlık: `v1.0.0 — SAUCIS revision reproducibility package`

Açıklamayı `RELEASE_NOTES_v1.0.0.md` içinden al. Tag'in lisans/metadata dahil
**son kontrol edilmiş commit** üzerinde olduğundan emin ol. Zenodo kaynak kod
arşivini bu commit'ten alır; yalnız release eklerine koymakla yetinme.

## 7. Zenodo kaydını kontrol et

İşlendikten sonra ad, ORCID, başlık, sürüm, lisans ve arşiv dosyalarını kontrol et.
Makalede bu release'e ait **version DOI** kullanılmalı. Bütün sürümleri temsil eden
concept DOI ile sürüm DOI'sini karıştırma. DOI, bu pakette önceden oluşturulmadı.

DOI'yi daha sonra CITATION'a yazmak bir sonraki committe yapılabilir; yayımlanmış
tag'i sessizce başka commit'e taşıma. İlk arşivde DOI'nin kendi metadatamızda
yazmaması, Zenodo'nun DOI atamasına engel değildir.

## Manuscript ve ek PDF

`docs/supplementary_methods/SAUCIS_Supplementary_Methods_standalone.pdf`, dergiye
ayrı **Supplementary Methods** dosyası olarak yüklenebilir; ana makaleye yapıştırmak
zorunda değilsin. GitHub/Zenodo paketi dergiye ek dosya yüklemenin yerine geçmez.
Gerçek repo ve DOI oluşunca Availability paragrafını bu doğrulanmış bilgilerle
sonlandıracağız. Paket yayımlanmadan "publicly available" diye tamamlandı yazma.

## Önemli kaynak notu

`.zenodo.json` ile `CITATION.cff` birlikte bulunursa Zenodo metadata için
`.zenodo.json` dosyasına öncelik verir; GitHub CITATION'ı ayrıca kullanır.
Bu yüzden metadata komutu ikisini birlikte günceller.
https://help.zenodo.org/docs/github/describe-software/zenodo-json/
https://help.zenodo.org/docs/github/describe-software/citation-file/

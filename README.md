# Aygaz Goruntu Isleme Bitirme Projesi

Kaggle veriseti dosyası link(13GB): https://www.kaggle.com/datasets/rrebirrth/animals-with-attributes-2

## 1. Projenin Amacı
Projenin ana hedefi, belirli bir veri setinde sinir ağı kullanarak görüntü sınıflandırması yapmaktır. Ayrıca, modelin performansını çeşitli veri manipülasyonlarına karşı değerlendirmek ve görüntü işleme yöntemlerinin etkisini analiz etmektir.

## 2. Ana İşlem Adımları

### Veri Hazırlığı
- **Veri Seti Doğrulama**: Belirtilen dizinde veri setinin ve her sınıfa ait alt klasörlerin varlığı kontrol edilir.
- **Görüntülerin Yüklenmesi ve Ön İşleme**:
  - Görüntüler belirtilen boyutlarda (128x128 piksel) yeniden boyutlandırılır.
  - Görüntüler 0-1 aralığında normalize edilir.
- **Etiketleme**: Her sınıfa bir indeks atanır ve etiketler `one-hot encoding` formatına dönüştürülür.

### Veri Bölünmesi
- Veri seti %70 eğitim ve %30 test olmak üzere ikiye ayrılır.

### Veri Artırma
- Eğitim verisi üzerinde dönüşüm (rotasyon, kaydırma, yatay çevirme vb.) uygulanarak veri artırımı yapılır.

## 3. Modelin Tasarımı

### CNN Yapısı
- **Konvolüsyon Katmanları**: Görüntü özelliklerini öğrenmek için 3 adet konvolüsyon ve maksimum havuzlama katmanı kullanılır.
- **Yoğun Katmanlar**: Özellik haritalarını düzleştirip sınıflandırma yapan 2 yoğun katman bulunur.
- **Aktivasyon Fonksiyonları**:
  - Ara katmanlarda ReLU,
  - Çıkış katmanında softmax kullanılır (çok sınıflı sınıflandırma için).
- **Düzenleme (Dropout)**: Aşırı öğrenmeyi önlemek için yoğun katmanlarda dropout uygulanır.

## 4. Modelin Eğitimi ve Değerlendirilmesi
- Model, eğitim setiyle 20 epoch boyunca eğitilir.
- Performans, test veri seti kullanılarak değerlendirilir.

## 5. Görüntü Manipülasyonu ve Ek Testler
Eğitimden sonra modelin dayanıklılığı şu manipülasyonlarla test edilir:
1. **Manipüle Edilmiş Görüntüler**:
   - Parlaklık Artışı ve Azaltılması: Görüntülerin parlaklık değerleri değiştirilerek yeni bir test seti oluşturulur.
2. **Renk Sabitliği Uygulanmış Görüntüler**:
   - Gray World Algoritması: Renk sabitliği sağlamak için görüntülere algoritma uygulanır.

Her bir manipülasyon için model test edilir ve başarı oranları karşılaştırılır.

## 6. Çıktılar
Eğitim tamamlandıktan sonra:
- Model, belirtilen dizine `.h5` formatında kaydedilir.
- Orijinal test seti, manipüle edilmiş görüntüler ve renk sabitliği uygulanmış görüntüler için test başarı oranları bir dosyaya yazılır.


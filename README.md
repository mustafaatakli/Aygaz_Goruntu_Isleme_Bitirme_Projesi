# Aygaz Goruntu Isleme Bitirme Projesi

# 1. Projenin Amacı
Bu proje, belirli bir veri setindeki görüntüleri kullanarak bir sinir ağı modeli oluşturmayı, modeli farklı veri manipülasyonlarına karşı değerlendirmeyi ve modelin dayanıklılığını test etmeyi amaçlamaktadır. Manipülasyonlar ve renk sabitliği gibi görüntü işleme yöntemlerinin model performansı üzerindeki etkisi analiz edilerek, derin öğrenme modellerinin veri setindeki değişikliklere karşı dayanıklılığı ve doğruluğunu artırmaya yönelik bir çerçeve sunulmaktadır.

---

# 2. Ana İşlem Adımları
## Veri Hazırlığı
Veri Seti Doğrulama: Belirtilen dizinde veri setinin ve her sınıfa ait alt klasörlerin varlığı kontrol edilir.
### Görüntülerin Yüklenmesi ve Ön İşleme:
Görüntüler belirtilen boyutlarda yeniden boyutlandırılır (128x128 piksel).
Görüntüler 0-1 aralığında normalize edilir.

Etiketleme: Her sınıfa bir indeks atanır ve etiketler one-hot encoding formatına dönüştürülür.
### Veri Bölünmesi
Veri seti, eğitim ve test veri seti olmak üzere %70 eğitim - %30 test oranında ayrılır.
### Veri Artırma
Eğitim verisi üzerinde dönüşüm (rotasyon, kaydırma, yatay çevirme vb.) uygulanarak veri artırımı yapılır.

---

# 3. Modelin Tasarımı
## CNN Yapısı: Model, görüntü sınıflandırma için bir Convolutional Neural Network (CNN) ile oluşturulur.
Konvolüsyon Katmanları: Görüntü özelliklerini öğrenmek için 3 adet konvolüsyon ve maksimum havuzlama bloğu kullanılır.
Yoğun Katmanlar: Özellik haritalarını düzleştirip sınıflandırma yapan 2 yoğun katman bulunur.
### Aktivasyon Fonksiyonları:
Ara katmanlarda ReLU,
Çıkış katmanında softmax kullanılır (çok sınıflı sınıflandırma için).
Düzenleme (Dropout): Aşırı öğrenmeyi önlemek için yoğun katmanlarda dropout uygulanır.

---

# 4. Modelin Eğitimi ve Değerlendirilmesi
Model, eğitim setiyle 20 epoch boyunca eğitilir.
Performans test veri seti kullanılarak değerlendirilir.

---

# 5. Görüntü Manipülasyonu ve Ek Testler
Eğitimden sonra modelin dayanıklılığı şu manipülasyonlarla test edilir:
## 1.Manipüle Edilmiş Görüntüler:
Parlaklık Artışı ve Azaltılması: Görüntülerin parlaklık değerleri değiştirilerek yeni bir test seti oluşturulur.
## 2.Renk Sabitliği Uygulanmış Görüntüler:
Gray World Algoritması: Renk sabitliği sağlamak için görüntülere algoritma uygulanır.
Her bir manipülasyon için model test edilir ve başarı oranları karşılaştırılır.

---

# 6. Çıktılar
## Eğitim tamamlandıktan sonra:
Model, belirtilen dizine .h5 formatında kaydedilir.
Orijinal test seti, manipüle edilmiş görüntüler ve renk sabitliği uygulanmış görüntüler için test başarı oranları bir dosyaya yazılır.


# Logistic Regression Karşılaştırması: Python ve Scikit-learn

## Proje Açıklaması

Bu proje, **Logistic Regression** algoritmasının iki farklı şekilde uygulanarak karşılaştırılmasını içermektedir. **Python**'da manuel olarak yazılan bir implementasyon ve **Scikit-learn** kütüphanesinin sunduğu **LogisticRegression** sınıfı ile gerçekleştirilen bir model karşılaştırılmaktadır. Proje, **heart.csv** veri setini kullanarak kalp hastalığı tahmini yapmaktadır. Veri seti, bireylerin sağlık bilgilerini kullanarak kalp hastalığına sahip olup olmadıklarını tahmin etmeyi amaçlamaktadır.

### Kullanılan Yöntemler

- **Manuel Logistic Regression (Python)**: 
   - Bu implementasyonda, **sigmoid** fonksiyonu ve **log-loss** (cross-entropy) fonksiyonu kullanılarak, modelin eğitilmesi ve test edilmesi sağlanmıştır. Eğitimde, ağırlıklar **gradient descent** algoritması ile güncellenmiştir.
   
- **Scikit-learn Logistic Regression**:
   - Bu implementasyonda ise, **Scikit-learn**'ün sunduğu **LogisticRegression** sınıfı kullanılarak, modelin eğitilmesi sağlanmıştır. Kütüphanenin sunduğu optimize edilmiş algoritmalar, daha hızlı ve verimli sonuçlar elde edilmesine yardımcı olmuştur.

## Veri Seti

Veri seti, **heart.csv** dosyasından alınmıştır. Bu veri seti, 13 özellik (feature) ve bir hedef etiket (target) içermektedir. **target** etiketi, bireyin kalp hastalığına sahip olup olmadığını (1: Hasta, 0: Sağlıklı) belirtmektedir.

### Özellikler:
- **age**: Yaş (Sürekli)
- **sex**: Cinsiyet (Kategorik)
- **cp**: Göğüs ağrısı tipi (Kategorik)
- **trestbps**: Dinlenme kan basıncı (Sürekli)
- **chol**: Kolesterol seviyesi (Sürekli)
- **fbs**: Açlık kan şekeri (Kategorik)
- **restecg**: Dinlenme elektrokardiyografi sonuçları (Kategorik)
- **thalach**: Maksimum kalp atış hızı (Sürekli)
- **exang**: Egzersiz kaynaklı anjina (Kategorik)
- **oldpeak**: ST depresyonu (Sürekli)
- **slope**: ST segmentinin eğimi (Kategorik)
- **ca**: Ana damar sayısı (Kategorik)
- **thal**: Talasemi türü (Kategorik)
- **target**: Kalp hastalığı var mı? (0: Hayır, 1: Evet)

### Veri Seti Temizliği:
Veri seti, eksik veri ve anormal değerler açısından kontrol edilmiştir. Eksik veri bulunmamakta ve veri seti düzgün bir şekilde düzenlenmiştir.

## Model Performansı ve Değerlendirme

### Karmaşıklık Matrisi

Karmaşıklık matrisi, modelin doğruluk, yanlış pozitif ve yanlış negatif tahminlerini görselleştirmemize olanak tanır. Karmaşıklık matrisini görselleştirerek, her iki modelin performansını değerlendirebiliriz.

**Manuel Logistic Regression Karmaşıklık Matrisi**:

```
[[100 0]
 [83 22]]
```

**Scikit-learn Logistic Regression Karmaşıklık Matrisi**:

``` 
[[79 21]
 [12 93]]
```

**Yorum**: Scikit-learn modeli daha az hata oranı ile daha iyi sonuçlar elde etmiştir. Özellikle yanlış negatif (False Negative) oranı daha düşüktür, bu da modelin gerçek kalp hastalarını daha doğru şekilde tespit ettiği anlamına gelir.

### Eğitim ve Test Süreleri

Eğitim ve test süreleri, modelin ne kadar hızlı çalıştığını ve veriyi ne kadar çabuk işlediğini gösterir.

**Manuel Logistic Regression**:
- Eğitim Süresi: 0.0056 saniye
- Test Süresi: 0.0023 saniye

**Scikit-learn Logistic Regression**:
- Eğitim Süresi: 0.0021 saniye
- Test Süresi: 0.0005 saniye

**Yorum**: Scikit-learn kütüphanesinin optimizasyon teknikleri, modelin eğitim ve test sürelerini önemli ölçüde hızlandırmıştır. Manuel implementasyon, optimizasyon eksikliği nedeniyle daha yavaş çalışmaktadır.

### Performans Değerlendirme Metrikleri

**Doğruluk (Accuracy)**, **Precision**, **Recall**, ve **F1-Score** gibi metrikler kullanılmıştır. Bu metrikler, modelin doğruluğu ve sınıflandırma başarısını daha ayrıntılı bir şekilde ölçmemize yardımcı olmuştur.

#### Manuel Logistic Regression:
```
Doğruluk: 0.60
Precision: 1
Recall: 0.21
F1-Score: 0.35
```

#### Scikit-learn Logistic Regression:
```
Doğruluk: 0.84
Precision: 0.82
Recall: 0.89
F1-Score: 0.85
```

**Yorum**: Scikit-learn modelinin doğruluğu ve diğer metrikleri manuel modele göre daha yüksektir. Özellikle **Recall** ve **Precision** değerleri, Scikit-learn modelinin kalp hastalıklarını tespit etme başarısını gösterir.

### Değerlendirme Metriklerinin Seçimi

Değerlendirme metrikleri seçiminde, **sınıf dağılımı** ve **problem türü** önemlidir. Veri setindeki **target** sınıfının dağılımı dengelidir (yaklaşık 50-50). Bu nedenle **accuracy** metriği geçerlidir. Ancak sınıflar arasında büyük dengesizlik varsa accuracy yanıltıcı olabilir. Bu durumlarda **Precision**, **Recall**, ve **F1-Score** gibi metriklerin de değerlendirilmesi gerekmektedir.

**Sınıf Dağılımı ve Değerlendirme Metrikleri**:
- Eğer sınıflar arasında dengesizlik varsa, sadece accuracy yerine **F1-Score** gibi daha dengeli metrikler kullanılmalıdır.
- Dengesiz veri setlerinde **Precision** ve **Recall** dikkatle incelenmelidir, çünkü yüksek accuracy değeri, düşük Recall veya Precision ile yanıltıcı olabilir.

## Sonuçlar ve Tartışma

- **Manuel Logistic Regression** ile yapılan modelde, optimizasyon ve verimlilik eksiklikleri nedeniyle eğitim süresi daha uzun ve doğruluk oranı daha düşük olmuştur.
- **Scikit-learn Logistic Regression** ise daha hızlı çalışmakta ve daha yüksek doğruluk oranlarına ulaşmaktadır. Özellikle kütüphanenin sunduğu optimizasyon teknikleri ve hata düzeltme yöntemleri sayesinde daha stabil sonuçlar elde edilmiştir.
- **Değerlendirme metrikleri**, veri setindeki sınıf dengesine bağlı olarak seçilmelidir. Bu çalışmada sınıf dengeli olduğu için accuracy geçerli bir metrik olarak kullanılabilmiştir.

### Kaynakça
- [https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

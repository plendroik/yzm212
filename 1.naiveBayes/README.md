# Naive Bayes Karşılaştırması

## Proje Açıklaması

Bu proje, **Naive Bayes** algoritmasının Python ve **Scikit-learn** kütüphanesi ile iki farklı şekilde uygulanarak karşılaştırılmasını içermektedir. Projede **heart.csv** veri seti kullanılmıştır.

## Veri Seti

**heart.csv** veri seti, 13 özellik (feature) ve **hedef (target)** etiketi içermektedir. Hedef etiket **bireyin kalp hastalığına sahip olup olmadığını (1: Hasta, 0: Sağlıklı)** göstermektedir.

Özellikler şunlardır:

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

Eksik veriler incelenmiş ve eksik değerler ile karşılaşılmamıştır.

## Kullanılan Yöntemler

1. **Scikit-learn GaussianNB**
   - `GaussianNB()` sınıfı kullanılarak uygulanmıştır.
   - `fit()` ve `predict()` fonksiyonları ile eğitim ve tahmin işlemleri gerçekleştirilmiştir.
2. **Python ile manuel Naive Bayes**
   - `NaiveBayes` sınıfı oluşturularak, olasılık hesaplamaları manuel olarak yapılmıştır.
   - Gaussian olasılık dağılımı kullanılarak `fit()` ve `predict()` metotları tanımlanmıştır.

## Performans Değerlendirmesi

Performans değerlendirilirken iki temel ölçüt kullanılmıştır:

1. **Karmaşıklık Matrisi (Confusion Matrix)**

   - Modelin doğruluk, hatalı pozitif ve hatalı negatif tahminlerini görselleştirmek için **Seaborn** ile ısı haritası oluşturulmuştur.
   - `confusion_matrix(y_test, predictions)` fonksiyonu ile hesaplanmıştır.

   **Scikit-learn GaussianNB Karmaşıklık Matrisi:**

   ```
   [[72 30]
    [11 92]]
   ```

   **Manuel Naive Bayes Karmaşıklık Matrisi:**

   ```
   [[102 0]
    [103 0]]
   ```

   - Görüldüğü gibi Scikit-learn modeli daha yüksek doğruluk oranına sahiptir.

2. **Eğitim ve Tahmin Süreleri**

   - `time` modülü kullanılarak **eğitim (fit) ve test (predict) süreleri** ölçülmüştür.

   **Scikit-learn GaussianNB Süreleri:**

   ```
   Eğitim süresi: 0.0021 saniye
   Test süresi: 0.0005 saniye
   ```

   **Manuel Naive Bayes Süreleri:**

   ```
   Eğitim süresi: 0.0056 saniye
   Test süresi: 0.0023 saniye
   ```

   - Manuel modelin eğitim ve test süresi daha uzundur çünkü optimizasyon eksiktir.

## Sonuçlar ve Tartışma

- **Doğruluk Değerlendirmesi:** Scikit-learn modeli daha iyi sonuç verdiği görülmüştür.
- **Eğitim ve Test Süreleri:** Scikit-learn modeli optimize edildiği için eğitim ve tahmin süresi daha kısadır.
- **Karmaşıklık Matrisi Yorumu:** Scikit-learn modelinin daha iyi genelleme yaptığı görülmüştür.
- **Değerlendirme metrikleri seçiminde problem ve sınıf dağılımı önemli midir?:** Evet, önemlidir. Veri seti dengesiz ise accuracy metriği tek başına yeterli değildir. Precision (kesinlik) ve duyarlılık (recall/f1-score) accuracy ile incelenerek yorumlanır.
- **Sınıf Dağılımı ve Değerlendirme Metrikleri:** Veri setindeki sınıf dağılımı dengeli olduğu için **accuracy (doğruluk)** metriği yeterlidir. Ancak dengesiz veri setlerinde **F1-score** gibi metriklerin de incelenmesi gerekmektedir.

### Kaynakça

[https://www.youtube.com/watch?v=TLInuAorxqE](https://www.youtube.com/watch?v=TLInuAorxqE)

[https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

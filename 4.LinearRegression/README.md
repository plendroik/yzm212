# Lineer Regresyon Model Karşılaştırması

Bu projede, bir öğrenci başarı verisi üzerinde iki farklı lineer regresyon modeli eğitilerek performansları karşılaştırılmıştır.

Kullanılan veri seti:  
Student Performance (Multiple Linear Regression) – Kaggle  
https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression

## Veri Kümesi Özellikleri

Veri kümesi, öğrencilerin ders başarısını etkileyebilecek bazı faktörleri içermektedir. Özellikler şu şekildedir:

- **Hours Studied**: Öğrencinin sınavdan önce çalıştığı saat sayısı (sayısal)
- **Previous Scores**: Öğrencinin önceki sınavlardan aldığı ortalama not (sayısal)
- **Performance Index**: Öğrencinin performans seviyesi (bağımlı değişken / hedef değişken)

Bu çalışma, "Hours Studied" ve "Previous Scores" bağımsız değişkenlerini kullanarak "Performance Index" değerini tahmin etmeyi amaçlamaktadır.

## Dosya Açıklamaları

| Dosya Adı                     | Açıklama                                                       |
|------------------------------|----------------------------------------------------------------|
| `LinearRegressionWSLearn.py` | Scikit-learn kütüphanesi kullanılarak eğitilen lineer regresyon modeli |
| `LinearRegressionWLSE.py`    | NumPy ile sıfırdan (from scratch) yazılmış least squares (en küçük kareler) çözümünü içeren model |

## Karşılaştırma Sonuçları

| Ölçüt                        | Scikit-learn Modeli       | From-Scratch (Least Squares) Modeli |
|------------------------------|---------------------------|-------------------------------------|
| Sabit Terim (Bias)           | -29.8168                  | -29.8168                            |
| Katsayılar                   | [2.8576, 1.0191]           | [2.8576, 1.0191]                     |
| Ortalama Kare Hata (MSE)     | 5.2143                    | 5.2143                              |
| Kök Ortalama Kare Hata (RMSE)| 2.2835                    | 2.2835                              |
| R² Skoru                     | 0.9859                    | 0.9859                              |

## Yorumlar

- Her iki model de aynı tahmin sonuçlarını üretmiştir.
- Scikit-learn kütüphanesi ile eğitilen modelin çıktıları, matematiksel olarak en küçük kareler yöntemiyle elde edilen from-scratch model ile birebir örtüşmektedir.
- Bu durum, her iki yöntemin de aynı kapalı formüle dayandığını göstermektedir:

  \[
  \theta = (X^T X)^{-1} X^T y
  \]

Bu da manuel olarak gerçekleştirilen modelleme sürecinin doğru uygulandığını ve Scikit-learn kütüphanesiyle sağlanan otomatik optimizasyonla aynı sonucu verdiğini kanıtlamaktadır.

## Sonuç

Bu çalışma, lineer regresyon modelinin hem hazır bir makine öğrenmesi kütüphanesi (Scikit-learn) hem de matematiksel temellere dayanan bir manuel çözüm (Least Squares) ile başarılı şekilde uygulanabileceğini göstermiştir. Model doğruluğu, iki yöntemin de güvenilir olduğunu ortaya koymaktadır.
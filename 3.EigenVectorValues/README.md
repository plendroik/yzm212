
# Matrisler, Özdeğerler ve Makine Öğrenmesi

Bu rapor, özdeğer-özvektör kavramları ile matris işlemlerinin makine öğrenmesindeki yeri, NumPy `eig` fonksiyonunun detayları ve manuel özdeğer hesaplama örneği ile `numpy.linalg.eig` karşılaştırmasını içermektedir.

## Soru 1: Makine Öğrenmesi, Matrisler ve Özdeğer İlişkisi

### Tanımlar

- **Makine Öğrenmesi**: Verilerden örüntüleri öğrenerek karar verme süreçlerini otomatikleştiren yapay zeka dalıdır.
- **Matrisler**: Sayıların iki boyutlu dizilişidir; verileri temsil etmek ve dönüştürmek için kullanılır.
- **Özdeğer ve Özvektör**: Bir matrisle çarpıldığında yalnızca ölçeklenen vektörler ve bu ölçekleme katsayısıdır.

### İlişki

Makine öğrenmesinde matrisler, verilerin gösterimi ve dönüşümü için kullanılır. Özellikle boyut indirgeme ve veri sıkıştırma gibi alanlarda özdeğerler ve özvektörler kritik rol oynar:

- **PCA (Principal Component Analysis)**: Kovaryans matrisinin özdeğer ayrışımı ile çalışır.
- **Lineer Transformasyonlar**: Özvektörler yönleri, özdeğerler büyüklükleri temsil eder.
- **SVD (Singular Value Decomposition)** ve **Kernel PCA** gibi yöntemler de özdeğer hesaplamasına dayanır.

### Kaynaklar

- https://machinelearningmastery.com/introduction-matrices-machine-learning/
- https://machinelearningmastery.com/introduction-to-eigendecomposition-eigenvalues-and-eigenvectors/ 

---

## Soru 2: NumPy `eig` Fonksiyonu İncelemesi

### Belge ve Kaynaklar

- Dokümantasyon: https://numpy.org/doc/2.1/reference/generated/numpy.linalg.eig.html  
- Kaynak kod: https://github.com/numpy/numpy/tree/main/numpy/linalg 

### Kullanımı

```python
import numpy as np

A = np.array([[4, 1], [2, 3]])
eigenvalues, eigenvectors = np.linalg.eig(A)
```

- `eigenvalues`: Özdeğerleri döndürür.
- `eigenvectors`: Sütunlarında her özdeğere karşılık gelen özvektörler vardır.

### İç Yapı

`eig` fonksiyonu arka planda LAPACK kütüphanelerini kullanır. Bu, C dilinde optimize edilmiş lineer cebir fonksiyonlarıdır. Özellikle `zgeev` ve `dgeev` fonksiyonları çağrılır.

---

## Soru 3: Manuel Özdeğer Hesaplama ve NumPy Karşılaştırması

### Kullanılan Matris

```python
matris = [
    [4, 1, 2, 3],
    [1, 2, 0, 1],
    [2, 0, 3, 1],
    [3, 1, 1, 4]
]
```

### Manuel Hesaplama
### Kaynak 
- Kaynak kod: https://github.com/LucasBN/Eigenvalues-and-Eigenvectors
#### Özdeğer

```python
ozdeger = ozdegerHesapla(matris)
print("Özdeğer: ", ozdeger)
# Çıktı: Özdeğer: 8.200000000000003
```

#### Özvektör

```python
ozvektor = ozvektorHesapla(matris, ozdeger)
print("Özvektör: ", ozvektor)
# Çıktı: Özvektör: None
```

### NumPy `eig` Sonuçları

```python
eigvals, eigvecs = np.linalg.eig(matris_np)
print("Özdeğerler:", eigvals)
print("Özvektörler:
", eigvecs)
```

#### Çıktı:

```
Özdeğerler: [8.19973759 0.64947433 2.67870395 1.47208412]

Özvektörler:
[[ 0.6649  0.7417 -0.0715 -0.0501]
 [ 0.2061 -0.1950  0.4759 -0.8323]
 [ 0.3736 -0.4276 -0.7826 -0.2547]
 [ 0.6130 -0.4783  0.3946  0.4895]]
```

### Karşılaştırma

| Yöntem           | Özdeğerler                         | Özvektörler       |
|------------------|-------------------------------------|-------------------|
| Manuel Hesaplama | 8.200000000000003                  | None              |
| NumPy `eig`      | [8.1997, 0.6494, 2.6787, 1.4721]    | Var (matris)      |

### Değerlendirme

- NumPy `eig` daha doğru ve kararlıdır.
- Manuel çözüm öğretici olsa da hesaplama ve doğruluk açısından sınırlıdır.

---

## Sonuç

Bu çalışmada, özdeğer-özvektör kavramlarının matematiksel ve uygulamalı temelleri ele alınmıştır. NumPy gibi bilimsel kütüphaneler ile bu kavramlar hızlı ve doğru biçimde uygulanabilir hale gelmektedir. Manuel yöntem ise süreci öğretici kılmak için önemlidir.
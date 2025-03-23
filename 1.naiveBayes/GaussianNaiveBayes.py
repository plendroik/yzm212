import numpy as np
import pandas as pd
import time 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

class NaiveBayes:
    
    def fit(self, X, y):
        ornekSayisi, ozellikSayisi = X.to_numpy().shape
        self._classes = np.unique(y)
        sinifSayisi = len(self._classes)

        self._mean = np.zeros((sinifSayisi, ozellikSayisi), dtype = np.float64)
        self._var = np.zeros((sinifSayisi, ozellikSayisi), dtype = np.float64)
        self._priors = np.zeros(sinifSayisi, dtype = np.float64)

        for indeks, sinif in enumerate(self._classes):
            X_sinif = X[y==sinif]
            self._mean[indeks, :] = X_sinif.to_numpy().mean(axis=0)
            self._var[indeks, :] = X_sinif.to_numpy().var(axis=0)
            self._priors[indeks] = X_sinif.to_numpy().shape[0] / float (ornekSayisi)
    def predict(self, X):
        X_np= X.to_numpy()
        y_tahmin = [self._predict(x) for x in X_np]
        return np.array(y_tahmin)
    def _predict(self, x):
        sonsalOlasiliklar = []
        for indeks, sinif in enumerate(self._classes):
            onselOlasilik = np.log(self._priors[indeks])
            sonsalOlasilik = np.sum(np.log(self._pdf(indeks, x)))
            sonsalOlasilik = sonsalOlasilik + onselOlasilik
            sonsalOlasiliklar.append(sonsalOlasilik)

        return self._classes[np.argmax(sonsalOlasilik)]
    def _pdf(self, sinif_indeks, x):
        mean = self._mean[sinif_indeks]
        var = self._var[sinif_indeks]
        bolunen = np.exp(-((x - mean) ** 2) / (2 * var))
        bolen = np.sqrt(2 * np.pi * var)
        return bolunen / bolen

dataFrame = pd.read_csv(r"yzm212\1.naiveBayes\heart.csv")

X = dataFrame.drop(columns=["target"])
y = dataFrame["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=61)

nbModel = NaiveBayes()

baslangic = time.time()
nbModel.fit(X_train, y_train)
egitimSuresi = time.time() - baslangic

baslangic = time.time()
predictions = nbModel.predict(X_test)
testSuresi = time.time() - baslangic

accuracy = accuracy_score(y_test, predictions)
print(f"Gaussian NB Model Doğruluğu: {accuracy:.2f}")
print(f"Eğitim süresi: {egitimSuresi:.4f} saniye")
print(f"Test süresi: {testSuresi:.4f} saniye")

karmasasiklikMatrisi = confusion_matrix(y_test, predictions)

plt.figure(figsize=(6, 4))
sns.heatmap(karmasasiklikMatrisi, annot=True, fmt="d", cmap="Reds", xticklabels=["Sağlıklı", "Hasta"], yticklabels=["Sağlıklı", "Hasta"])
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.title("Karmaşıklık Matrisi")
plt.show()
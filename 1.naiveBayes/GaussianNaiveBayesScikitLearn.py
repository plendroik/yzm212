import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

dataFrame = pd.read_csv(r"yzm212\1.naiveBayes\heart.csv")

X = dataFrame.drop(columns=["target"])
y = dataFrame["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=61)

nbModel = GaussianNB()

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
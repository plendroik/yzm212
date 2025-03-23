import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import time

df = pd.read_csv(r"yzm212\2.logisticRegression\heart.csv")

X = df.drop(columns=["target"])
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=61, stratify=y)

logistikModel = LogisticRegression()

baslangic = time.time()
logistikModel.fit(X_train, y_train)
egitimSuresi = time.time() - baslangic

baslangic = time.time()
y_pred = logistikModel.predict(X_test)
testSuresi = time.time() - baslangic

print("Scikitlearn Model Performansı:")
print("Karmaşıklık matrisi:\n", confusion_matrix(y_test, y_pred))
print("Sınıflandırma raporu:\n", classification_report(y_test, y_pred))
print("Doğruluk:", accuracy_score(y_test, y_pred))
print(f"Eğitim Süresi: {egitimSuresi} saniye")
print(f"Test Süresi: {testSuresi} saniye")
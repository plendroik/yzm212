import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import time

class LogistikRegresyon:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        z = np.clip(z, -500, 500) # overflow önlemek için sınır belirliyorum
        return 1 / (1 + np.exp(-z))

    def compute_cost(self, X, y):
        m = X.shape[0]
        z = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(z)
        epsilon = 1e-10
        cost = -np.mean(y * np.log(y_pred+epsilon) + (1 - y) * np.log(1 - y_pred+epsilon))# log0 olmasın diye çok küçük bir sayı olan epsilon ekliyorum
        return cost

    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)  
        self.bias = 0  

        for _ in range(self.n_iterations):
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(z)

            dw = np.dot(X.T, (y_pred - y)) / m
            db = np.mean(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            if _ % 100 == 0:
                print(f"Cost is {self.compute_cost(X, y)}")

    def predict(self, X, threshold=0.5):
        z = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(z)
        return (y_pred >= threshold).astype(int)

df = pd.read_csv(r"yzm212\2.logisticRegression\heart.csv")

X = df.drop(columns = ["target"])
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2 , random_state = 61, stratify = y)

logistikModel = LogistikRegresyon(learning_rate=0.01, n_iterations=1000)

baslangic = time.time()
logistikModel.fit(X_train, y_train)
egitimSuresi = time.time() - baslangic

baslangic = time.time()
y_pred = logistikModel.predict(X_test)
testSuresi = time.time() - baslangic

print("Bayes Model Performansı:")
print("Karmaşıklık matrisi:\n", confusion_matrix(y_test, y_pred))
print("Sınıflandırma raporu:\n", classification_report(y_test, y_pred))
print("Doğruluk:", accuracy_score(y_test, y_pred))
print(f"Eğitim Süresi: {egitimSuresi} saniye")
print(f"Test Süresi: {testSuresi} saniye")
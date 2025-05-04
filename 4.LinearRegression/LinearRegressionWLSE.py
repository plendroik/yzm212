import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

df = pd.read_csv(r"yzm212\4.LinearRegression\Student_Performance.csv")

X = df[["Hours Studied", "Previous Scores"]].values
y = df["Performance Index"].values.reshape(-1, 1)

X_b = np.c_[np.ones((X.shape[0], 1)), X] # bias için birler ekle

# θ = (X^T * X)^-1 * X^T * y
theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

y_pred = X_b.dot(theta)

print("Theta (Bias + Coefficients):", theta.ravel())

# karşılaştırma ölçütleri
mse = np.mean((y_pred - y)**2)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

print("MSE:", mse)
print("RMSE:", rmse)
print("R² Score:", r2)

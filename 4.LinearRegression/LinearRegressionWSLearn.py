import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv(r"yzm212\4.LinearRegression\Student_Performance.csv")

X = df[["Hours Studied", "Previous Scores"]].values
y = df["Performance Index"].values.reshape(-1, 1)

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)


print("Intercept (Bias):", model.intercept_[0])
print("Coefficients:", model.coef_[0])

# karşılaştırma ölçütleri
mse = np.mean((y_pred - y)**2)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

print("MSE:", mse)
print("RMSE:", rmse)
print("R² Score:", r2)
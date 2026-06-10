import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("car data.csv")

print("First 5 Rows:")
print(df.head())

# -----------------------------
# 2. Dataset Information
# -----------------------------
print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

# -----------------------------
# 3. Feature Engineering
# -----------------------------
current_year = 2024

df["Car_Age"] = current_year - df["Year"]

# Drop unnecessary columns
df.drop(["Car_Name", "Year"], axis=1, inplace=True)

# -----------------------------
# 4. Convert Categorical Data
# -----------------------------
df = pd.get_dummies(df, drop_first=True)

print("\nProcessed Columns:")
print(df.columns)

# -----------------------------
# 5. Correlation Heatmap
# -----------------------------
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# -----------------------------
# 6. Features and Target
# -----------------------------
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# 7. Train Model
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# 8. Predictions
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# 9. Model Evaluation
# -----------------------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print("MAE :", round(mae, 2))
print("MSE :", round(mse, 2))
print("RMSE :", round(rmse, 2))
print("R² Score :", round(r2, 2))

# -----------------------------
# 10. Actual vs Predicted
# -----------------------------
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")

plt.show()

# -----------------------------
# 11. Feature Importance
# -----------------------------
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

coefficients = coefficients.sort_values(
    by="Coefficient",
    ascending=False
)

print("\nFeature Importance:")
print(coefficients)

plt.figure(figsize=(10, 6))
sns.barplot(
    x="Coefficient",
    y="Feature",
    data=coefficients
)

plt.title("Feature Importance")
plt.show()

# -----------------------------
# 12. Sample Prediction
# -----------------------------
sample_car = X.iloc[0:1]

predicted_price = model.predict(sample_car)

print(
    "\nPredicted Price for Sample Car:",
    round(predicted_price[0], 2),
    "Lakhs"
)

print("\nProject Completed Successfully!")
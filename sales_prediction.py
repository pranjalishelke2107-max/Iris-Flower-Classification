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
df = pd.read_csv("advertising.csv")

print("First 5 Rows:")
print(df.head())

# -----------------------------
# 2. Data Cleaning
# -----------------------------
if "Unnamed: 0" in df.columns:
    df.drop("Unnamed: 0", axis=1, inplace=True)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDataset Shape:")
print(df.shape)

# -----------------------------
# 3. Statistical Summary
# -----------------------------
print("\nStatistics:")
print(df.describe())

# -----------------------------
# 4. Correlation Analysis
# -----------------------------
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# -----------------------------
# 5. Advertising vs Sales
# -----------------------------
plt.figure(figsize=(6,4))
sns.scatterplot(x="TV", y="Sales", data=df)
plt.title("TV Advertising vs Sales")
plt.show()

plt.figure(figsize=(6,4))
sns.scatterplot(x="Radio", y="Sales", data=df)
plt.title("Radio Advertising vs Sales")
plt.show()

plt.figure(figsize=(6,4))
sns.scatterplot(x="Newspaper", y="Sales", data=df)
plt.title("Newspaper Advertising vs Sales")
plt.show()

# -----------------------------
# 6. Feature Selection
# -----------------------------
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

# -----------------------------
# 7. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# 8. Train Model
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# 9. Prediction
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# 10. Model Evaluation
# -----------------------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nMODEL PERFORMANCE")
print("MAE :", round(mae, 2))
print("MSE :", round(mse, 2))
print("RMSE :", round(rmse, 2))
print("R² Score :", round(r2, 2))

# -----------------------------
# 11. Actual vs Predicted
# -----------------------------
plt.figure(figsize=(7,5))
plt.scatter(y_test, y_pred)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")

plt.show()

# -----------------------------
# 12. Feature Importance
# -----------------------------
importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print("\nAdvertising Impact")
print(importance)

plt.figure(figsize=(8,5))
sns.barplot(
    x="Coefficient",
    y="Feature",
    data=importance
)

plt.title("Impact of Advertising Channels")
plt.show()

# -----------------------------
# 13. Future Sales Prediction
# -----------------------------
new_campaign = pd.DataFrame({
    "TV": [250],
    "Radio": [40],
    "Newspaper": [50]
})

future_sales = model.predict(new_campaign)

print(
    "\nPredicted Sales for New Campaign:",
    round(future_sales[0], 2)
)

print("\nProject Completed Successfully!")
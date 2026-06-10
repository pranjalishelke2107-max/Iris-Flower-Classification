import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("unemployment.csv")

# Clean Column Names
df.columns = df.columns.str.strip()

# Convert Date Column
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

# Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Remove Duplicates
df.drop_duplicates(inplace=True)

# ---------------------------
# Unemployment Trend
# ---------------------------

plt.figure(figsize=(12,6))

monthly_unemployment = df.groupby('Date')[
    'Estimated Unemployment Rate (%)'
].mean()

monthly_unemployment.plot()

plt.title("Average Unemployment Rate Over Time")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.grid(True)

plt.show()

# ---------------------------
# Covid Impact Analysis
# ---------------------------

covid_period = df[df['Date'] >= '2020-03-01']

plt.figure(figsize=(12,6))

sns.lineplot(
    x='Date',
    y='Estimated Unemployment Rate (%)',
    data=covid_period
)

plt.title("Covid-19 Impact on Unemployment")
plt.xticks(rotation=45)

plt.show()

# ---------------------------
# Region-wise Analysis
# ---------------------------

state_unemployment = df.groupby('Region')[
    'Estimated Unemployment Rate (%)'
].mean().sort_values(ascending=False)

plt.figure(figsize=(12,8))

sns.barplot(
    x=state_unemployment.values,
    y=state_unemployment.index
)

plt.title("Average Unemployment Rate by Region")
plt.xlabel("Unemployment Rate (%)")
plt.ylabel("Region")

plt.show()

# ---------------------------
# Urban vs Rural
# ---------------------------

plt.figure(figsize=(8,5))

sns.boxplot(
    x='Area',
    y='Estimated Unemployment Rate (%)',
    data=df
)

plt.title("Urban vs Rural Unemployment")

plt.show()

# ---------------------------
# Monthly Trend
# ---------------------------

df['Month'] = df['Date'].dt.month

monthly_pattern = df.groupby('Month')[
    'Estimated Unemployment Rate (%)'
].mean()

plt.figure(figsize=(10,5))

sns.lineplot(
    x=monthly_pattern.index,
    y=monthly_pattern.values,
    marker='o'
)

plt.title("Seasonal Monthly Unemployment Trend")
plt.xlabel("Month")
plt.ylabel("Average Unemployment Rate (%)")

plt.show()

# ---------------------------
# Top 10 Regions
# ---------------------------

print("\nTop 10 Regions with Highest Unemployment:")
print(state_unemployment.head(10))

# ---------------------------
# Before vs After Covid
# ---------------------------

before_covid = df[df['Date'] < '2020-03-01']
after_covid = df[df['Date'] >= '2020-03-01']

print("\nAverage Unemployment Before Covid:")
print(before_covid['Estimated Unemployment Rate (%)'].mean())

print("\nAverage Unemployment During Covid:")
print(after_covid['Estimated Unemployment Rate (%)'].mean())

print("\nAnalysis Completed Successfully!")
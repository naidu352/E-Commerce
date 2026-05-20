
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ttest_1samp
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("ecommerce_sales_analysis.csv")

print("rows:", len(df))
print("columns:", len(df.columns))
print("\nmissing values:\n", df.isnull().sum())


df = df.drop_duplicates()
df.fillna(df.median(numeric_only=True), inplace=True)

df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
df['Month'] = df['Order Date'].dt.month


print(df.describe())
print(df.corr(numeric_only=True))

num_cols = ['Sales', 'Quantity', 'Profit']

print("\niqr outlier detection")
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(col, "outliers:", len(outliers))


for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df = df[(df[col] >= lower) & (df[col] <= upper)]

print("rows after cleaning:", len(df))


# objective 1
corr = df['Quantity'].corr(df['Sales'])
print("\ncorrelation (Quantity vs Sales):", corr)


# objective 2
print("\ncategory sales:\n", df.groupby('Category')['Sales'].sum())


# objective 3: visualization

# 1. Scatter Plot
plt.figure()
sns.scatterplot(x='Profit', y='Sales', data=df)
plt.title("Profit vs Sales")
plt.show()

# 2. Histogram
plt.figure()
sns.histplot(df['Sales'], kde=True)
plt.title("Sales Distribution")
plt.show()

# 3. Line Plot
monthly = df.groupby('Month')['Sales'].mean().reset_index()
plt.figure()
sns.lineplot(x='Month', y='Sales', data=monthly, marker='o')
plt.title("Monthly Sales Trend")
plt.show()

# 4. Bubble Plot 
plt.figure()
plt.scatter(df['Quantity'], df['Sales'], s=df['Profit']*2, alpha=0.5)
plt.title("Bubble Plot (Quantity vs Sales with Profit size)")
plt.xlabel("Quantity")
plt.ylabel("Sales")
plt.show()

# 5. Box Plot
plt.figure()
sns.boxplot(x='Region', y='Sales', data=df)
plt.title("Region vs Sales")
plt.show()

# 6. Bar Plot 
category_sales = df.groupby('Category')['Sales'].sum()
plt.figure()
category_sales.plot(kind='bar')
plt.title("Sales by Category")
plt.show()

# 7. Heatmap 
plt.figure()
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# 8. Pie Chart
plt.figure()
df['Category'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title("Category Distribution")
plt.ylabel('')
plt.show()


# objective 4
t_stat, p_val = ttest_1samp(df['Sales'], 500)
print("\nt-test p-value:", p_val)

mean = df['Sales'].mean()
std = df['Sales'].std()
n = len(df)

z = (mean - 500) / (std / np.sqrt(n))
print("z-score:", z)

# objective 5
x = df['Quantity']
y = df['Sales']

mean_x = x.mean()
mean_y = y.mean()

num = ((x - mean_x) * (y - mean_y)).sum()
den = ((x - mean_x) ** 2).sum()
print("hi")

slope = num / den
intercept = mean_y - slope * mean_x

print("\nregression equation:")
print("y =", slope, "* x +", intercept)

plt.figure()
plt.figure()
sns.regplot(x='Quantity', y='Sales', data=df)
plt.title("Regression Plot")
plt.show()

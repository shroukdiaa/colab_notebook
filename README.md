This project focuses on analyzing and predicting Canada’s per-capita income using
Linear Regression and Gradient Descent.
It includes data exploration, model building, evaluation, and visualization.

📁 Project Structure
│── canada_per_capita_income.csv
│── model_linear_regression.py
│── gradient_descent.py
│── visualization.ipynb
│── README.md

📌 Objective

To build a predictive model that estimates Canada’s per-capita income based on the year, using:

Linear Regression (Sklearn)

Gradient Descent (Implemented manually)

📊 Dataset Information

Rows: 47

Columns:

year

per capita income (US$)

No missing values

Data range: 1970 → 2016

🔍 Exploratory Data Analysis
Summary Statistics:

Mean per capita income: 18920.13 USD

Min: 3399.29 USD

Max: 42676.46 USD

Used Commands:
data.head()
data.info()
data.describe()
data.isnull().sum()

🤖 Model 1 — Linear Regression (Sklearn)
Code:
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = df[['year']]
Y = df['per capita income (US$)']

model = LinearRegression()
model.fit(X, Y)

pred_2020 = model.predict([[2020]])
print(pred_2020)

🧮 Model 2 — Gradient Descent (Manually Implemented)
Code:
X = df['year'].values
Y = df['per capita income (US$)'].values

X = (X - X.mean()) / X.std()
X = np.c_[np.ones(X.shape[0]), X]

b0 = 0
b1 = 0
learning_rate = 0.01
epochs = 1000

for _ in range(epochs):
    y_pred = X.dot([b0, b1])
    d_b0 = (-2/len(X)) * np.sum(Y - y_pred)
    d_b1 = (-2/len(X)) * np.sum((Y - y_pred) * X[:,1])
    b0 -= learning_rate * d_b0
    b1 -= learning_rate * d_b1

print("b0:", b0)
print("b1:", b1)

📈 Predictions
Example:
year_2020 = (2020 - df['year'].mean()) / df['year'].std()
predicted_income_2020 = b0 + b1 * year_2020

📉 Visualization
plt.scatter(df['year'], df['per capita income (US$)'])
plt.plot(df['year'], model.predict(df[['year']]), color='red')
plt.xlabel('Year')
plt.ylabel('Per Capita Income (US$)')
plt.show()

Sample Plot (Add your image here):

🏁 Results

Linear regression shows a strong positive trend

Gradient descent successfully approximates model parameters

Prediction for 2020 suggests continued income growth

📬 Contact

If you have suggestions or want to contribute, feel free to reach out!

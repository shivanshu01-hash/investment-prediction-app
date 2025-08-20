import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
dataset = pd.read_csv(r'C:\Users\dell\OneDrive\Desktop\FSDSAI\ML_Workspace\MLR\Investment.csv')

st.title("📊 Investment Prediction Showcase")

# Show dataset
st.subheader("🔹 Dataset")
st.dataframe(dataset.head(10))

# Split data
x = dataset.iloc[:, :-1]
y = dataset.iloc[:, 4]
x = pd.get_dummies(x, dtype=int)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Train model
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Prediction
y_pred = regressor.predict(x_test)

# Show coefficients
st.subheader("📌 Model Coefficients")
st.write("Coefficients:", regressor.coef_)
st.write("Intercept:", regressor.intercept_)

# Plot actual vs predicted with regression line
st.subheader("📈 Visualization")
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.7, label="Predicted Points")

# Regression line (ideal case: y = x)
line_start = min(y_test.min(), y_pred.min())
line_end = max(y_test.max(), y_pred.max())
ax.plot([line_start, line_end], [line_start, line_end], color="red", linewidth=2, label="Regression Line")

ax.set_xlabel("Actual Values")
ax.set_ylabel("Predicted Values")
ax.set_title("Actual vs Predicted with Regression Line")
ax.legend()
st.pyplot(fig)
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Import FILE
csv_path = "house_price_simple.csv"
df = pd.read_csv(csv_path, usecols=["area_m2", "price_million_vnd"])

# Clean DATA
df ["area_m2"] = pd.to_numeric(df["area_m2"], errors="coerce")
df["price_million_vnd"] = pd.to_numeric(df["price_million_vnd"], errors="coerce")
df = df.dropna()
if len(df) < 2:
    raise ValueError("Not enough valid data points after cleaning.")

# Prepare DATA
x = df[["area_m2"]].values
y = df["price_million_vnd"].values

# Train MODEL
model = LinearRegression()
model.fit(x, y)

# Make PREDICTIONS
raw = input("Enter area in m2: ").strip().replace(",", ".")
pred_val = float(model.predict(np.array([[float(raw)]]))[0])
print(f"Predicted price for {raw} m2: {pred_val:.2f} million VND")
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

#Import FILE
csv_path = "weather_train_sample.csv"
df = pd.read_csv(csv_path, usecols=["temp_c","wind_m_s","rain"])

#Clean DATA
df["temp_c"] = pd.to_numeric(df["temp_c"], errors='coerce')
df["wind_m_s"] = pd.to_numeric(df["wind_m_s"], errors='coerce')
df["rain"] = pd.to_numeric(df["rain"], errors='coerce')
df = df.dropna()
df = df[(df["rain"]==0) | (df["rain"]==1)]
if len(df) < 10:
    raise ValueError("Not enough valid data to train the model.")

#Prepare DATA
X = df[["temp_c", "wind_m_s"]]
y = df["rain"].values.astype(int)

#Train MODEL
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

#Make PREDICTION
raw_temp = input("Enter temperature in Celsius: ").strip().replace(",", ".")
raw_wind = input("Enter wind speed in m/s: ").strip().replace(",", ".")

temp_val = float(raw_temp)
wind_val = float(raw_wind)

prediction = model.predict_proba(np.array([[temp_val, wind_val]]))[:, 1][0]

threshold = 0.5
label = 1 if prediction >= threshold else 0
print(f"Predicted probability of rain: {prediction:.2f}")
print(f"Rain label (1 for rain, 0 for no rain): {label}")
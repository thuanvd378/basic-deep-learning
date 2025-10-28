import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

#Import FILE
csv_path = "apartment_rent_sample.csv"
df = pd.read_csv(csv_path, usecols=["area_m2", "distance_km", "rent_million_vnd"])

#Clean DATA
df["area_m2"] = pd.to_numeric(df["area_m2"], errors='coerce')
df["distance_km"] = pd.to_numeric(df["distance_km"], errors='coerce')
df["rent_million_vnd"] = pd.to_numeric(df["rent_million_vnd"], errors='coerce')
df = df.dropna()
if len(df) < 2:
    raise ValueError("Not enough valid data to train the model.")

#Prepare DATA 
X = df[["area_m2", "distance_km"]].values
y = df["rent_million_vnd"].values

#Train MODEL
model = LinearRegression()
model.fit(X, y)

#Make PREDICTION
raw_area = input("Enter apartment area in m2: ").strip().replace(',', '.')
raw_distance = input("Enter distance to city center in km: ").strip().replace(',', '.')
area_val = float(raw_area)
distance_val = float(raw_distance)
predicted_rent = model.predict(np.array([[area_val, distance_val]]))
print("Predicted rent : " , predicted_rent[0] , " million VND")

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

orders = pd.read_csv("data/delivery_times.csv")
FEATURES = ["distance_km", "prep_time_min", "traffic_level", "rain"]

X = orders[FEATURES]
y = orders["delivery_min"]

# Split FIRST, then scale -- fit the scaler only on training data to avoid leakage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)
mae = mean_absolute_error(y_test, model.predict(X_test_scaled))
print(f"Linear Regression test MAE: {mae:.2f} minutes")

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

orders = pd.read_csv("data/delivery_times.csv")
FEATURES = ["distance_km", "prep_time_min", "traffic_level", "rain"]

X = orders[FEATURES]
y = orders["delivery_min"]

# NOTE: scaling the whole dataset before splitting -- this is a bug we fix next commit
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
mae = mean_absolute_error(y_test, model.predict(X_test))
print(f"Linear Regression test MAE: {mae:.2f} minutes")

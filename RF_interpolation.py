import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the survey data into a pandas dataframe
survey_data = pd.read_csv("survey_data.csv")

# Select the input features and target variable
X = survey_data[["LATITUDE", "LONGITUDE"]]
y = survey_data["INCOME"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Set up the pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train the model using the pipeline
pipeline.fit(X_train, y_train)

# Evaluate the model using cross-validation
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')
print("Cross-Validation R² Scores:", cv_scores)
print("Average CV R² Score:", np.mean(cv_scores))

# Predict on the test set
y_pred = pipeline.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Metrics on Test Set:")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")

# Plotting Residuals
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), linestyles='dashed', colors='red')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

# Feature Importance
model = pipeline.named_steps['model']
feature_importances = model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
print("\nFeature Importances:")
print(importance_df)

# Generate random spatial points
n_points = 100
min_lat, max_lat = X["LATITUDE"].min(), X["LATITUDE"].max()
min_lon, max_lon = X["LONGITUDE"].min(), X["LONGITUDE"].max()
random_lats = np.random.uniform(min_lat, max_lat, n_points)
random_lons = np.random.uniform(min_lon, max_lon, n_points)
new_points = pd.DataFrame({"LATITUDE": random_lats, "LONGITUDE": random_lons})

# Use the trained model to predict INCOME value for the new points
predictions = pipeline.predict(new_points)

# Print the predicted INCOME values for the new points
print("\nPredicted income values for new points:")
print(predictions)

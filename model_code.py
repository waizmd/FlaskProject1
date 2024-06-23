import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
#import warnings

#warnings.filterwarnings("ignore")

# Hardcoded data
data = pd.DataFrame({
    'Age': [25, 28, 32, 35, 40, 45, 50, 55, 60, 30,
            27, 29, 33, 36, 41, 46, 51, 56, 59, 31,
            26, 34, 38, 42, 48, 53, 58, 37, 43, 47],
    'Experience': [3, 5, 7, 10, 15, 20, 25, 30, 35, 8,
                   4, 6, 9, 12, 17, 22, 27, 32, 34, 7,
                   2, 11, 13, 16, 23, 28, 33, 14, 18, 24],
    'Salary': [45000, 50000, 60000, 65000, 75000, 85000, 95000, 105000, 115000, 58000,
               47000, 52000, 63000, 67000, 77000, 87000, 97000, 107000, 112000, 61000,
               46000, 64000, 70000, 76000, 88000, 98000, 108000, 68000, 78000, 92000]
})

# Features and target
X = data[['Age', 'Experience']]
y = data['Salary']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create and train the Random Forest Regressor
regressor = RandomForestRegressor(n_estimators=100, random_state=0)
regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

# Display predictions
predictions = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})

print(predictions)

pickle.dump(regressor,open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))

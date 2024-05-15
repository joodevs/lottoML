import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

# Load the data
file_path = 'lottoHistory.xlsx'  # Update this to your actual file path
lotto_data = pd.read_excel(file_path)

# Define the columns for the draw numbers and the bonus number
draw_columns = ['First', 'Second', 'Third', 'Fourth', 'Fifth', 'Sixth']

# Initialize a DataFrame with zeros for one-hot encoding
one_hot_encoded_draws = pd.DataFrame(0, index=np.arange(len(lotto_data)), columns=np.arange(1, 46))

# Fill the DataFrame with 1 where the number was drawn
for col in draw_columns:
    for index, value in lotto_data[col].items():
        one_hot_encoded_draws.at[index, value] = 1

# Prepare features (X) and targets (y) for the model
X = one_hot_encoded_draws.values[:-1]  # All rows except the last
y = one_hot_encoded_draws.shift(-1).values[:-1]  # Shift rows up by one and exclude the last row

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Initialize the Random Forest model
model = RandomForestClassifier(random_state=42)

# Grid search with cross-validation
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Print best parameters
print(f"Best parameters: {grid_search.best_params_}")

# Evaluate the model using cross-validation
cv_scores = cross_val_score(grid_search.best_estimator_, X, y, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV score: {np.mean(cv_scores)}")

# Predict on the last observed draw (not future draw since it's unknown)
last_draw_observed = X_test[-1].reshape(1, -1)
predicted_probs = grid_search.predict_proba(last_draw_observed)

# Calculate the probabilities for each number being drawn
number_probs = np.array([prob[:, 1] for prob in predicted_probs]).T  # Take the probability of '1' (being drawn) for each number

# Identify the 6 numbers with the highest probabilities
top_6_numbers = np.argsort(-number_probs.ravel())[:6] + 1  # Add 1 because lottery numbers start from 1

# Display the predicted numbers
print(f"Predicted main draw numbers: {top_6_numbers}")
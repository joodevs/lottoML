import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load the data
file_path = 'lottoHistory.xlsx'  # Update this with your actual file path
lotto_data = pd.read_excel(file_path)

# Define the columns for the draw numbers and the bonus number
draw_columns = ['First', 'Second', 'Third', 'Fourth', 'Fifth', 'Sixth']
bonus_column = 'Bonus'  # Bonus number column

# Initialize a DataFrame with zeros for one-hot encoding
one_hot_encoded_draws = pd.DataFrame(0, index=np.arange(len(lotto_data)), columns=np.arange(1, 46), dtype=np.int8)

# Fill the DataFrame with 1 where the number was drawn
for col in draw_columns:
    for index, value in lotto_data[col].items():
        one_hot_encoded_draws.at[index, value] = 1

# Prepare features (X) and targets (y) for the model
X = one_hot_encoded_draws.iloc[:-1]  # All rows except the last
y = one_hot_encoded_draws.shift(-1).iloc[:-1]  # Shift rows up by one and exclude the last row

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict the next draw
predicted_probs = model.predict_proba(X_test.iloc[-1].values.reshape(1, -1))

# Calculate the probabilities for each number being drawn
number_probs = np.array([prob[:, 1] for prob in predicted_probs]).T  # Take the probability of '1' (being drawn) for each number

# Function to select a set of numbers based on their probabilities
def select_numbers(probabilities, exclude_numbers, top_n=6):
    sorted_indices = np.argsort(-probabilities)
    selected_numbers = []
    for idx in sorted_indices:
        if idx + 1 not in exclude_numbers:
            selected_numbers.append(idx + 1)
        if len(selected_numbers) == top_n:
            break
    return selected_numbers

# Select the top 6 numbers as the first set
top_6_numbers = select_numbers(number_probs.ravel(), [])

# Generate 4 additional sets
additional_sets = [top_6_numbers]
for i in range(1, 5):
    # Exclude numbers already chosen in previous sets
    exclude_numbers = {num for s in additional_sets for num in s}
    new_set = select_numbers(number_probs.ravel(), exclude_numbers)
    additional_sets.append(new_set)

# Display the predicted numbers
for i, number_set in enumerate(additional_sets, 1):
    print(f"Set {i}: {sorted(number_set)}")

# Calculate the frequency of the bonus number and predict the most likely bonus number
bonus_freq = lotto_data[bonus_column].value_counts()
predicted_bonus = bonus_freq.idxmax()

# Display the predicted bonus number
print(f"Predicted bonus number: {predicted_bonus}")
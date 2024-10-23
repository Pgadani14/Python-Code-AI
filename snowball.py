from nba_api.stats.endpoints import leaguedashplayerstats
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense
import matplotlib.pyplot as plt
from keras._tf_keras.keras.optimizers import Adam

stats_2024 = leaguedashplayerstats.LeagueDashPlayerStats(season="2023-24").get_data_frames()[0]
stats_2022 = leaguedashplayerstats.LeagueDashPlayerStats(season="2021-22").get_data_frames()[0]
stats_2023 = leaguedashplayerstats.LeagueDashPlayerStats(season="2022-23").get_data_frames()[0]
stats_2025 = leaguedashplayerstats.LeagueDashPlayerStats(season="2024-25").get_data_frames()[0]

stats_2022['Season'] = '2021-22'
stats_2023['Season'] = '2022-23'
stats_2024['Season'] = '2023-24'

all_seasons = pd.concat([stats_2022, stats_2023, stats_2024], ignore_index=True)

# Drop rows with missing data
all_seasons_df = all_seasons.dropna()

# Features (X) and Target (y)
# Prepare features and target
features = ['REB', 'AST', 'STL', 'BLK', 'TOV', 'GP', 'MIN', 'W', 'L', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'OREB', 'DREB', 'PF', 'PFD']
X = all_seasons[features]
#X = all_seasons_df[['REB', 'AST', 'STL', 'BLK', 'TOV', 'GP', 'MIN']] # Features (you can adjust this)
y = all_seasons_df['PTS'] # Target: Points per game

# Train-Test Split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features (important for some models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the model
model = Sequential()

# Input layer (7 features -> 128 neurons)
model.add(Dense(256, activation='relu', input_shape=(X_train_scaled.shape[1],)))

# Hidden layers (2 hidden layers with 128, 64 and 32 neurons)
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))

# Output layer (1 neuron for regression output - predicting points)
model.add(Dense(1, activation='linear'))

# Lower learning rate for fine-tuning
learning_rate = 0.001  # Try different values like 0.1, 0.01, 0.001, etc.
optimizer = Adam(learning_rate=learning_rate)

# Compile the model with the tuned learning rate
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

# Display the model summary
model.summary()

# Train the model
model.fit(X_train_scaled, y_train, epochs=150, batch_size=64, validation_split=0.2, verbose=1)

# Training complete
print("Training complete.")

# Evaluate the model on test data
test_loss, test_mae = model.evaluate(X_test_scaled, y_test)

print(f"Test Mean Absolute Error: {test_mae:.2f}")

# Now, predict points for all players in 'all_seasons'
X_all = all_seasons[features]
X_all_scaled = scaler.transform(X_all)
predicted_points = model.predict(X_all_scaled)
all_seasons['Predicted_PTS'] = predicted_points.flatten()
all_seasons['Predicted_PTS'] = all_seasons['Predicted_PTS'].round(2)

# Prepare data for export
export_columns = ['PLAYER_NAME', 'Season', 'PTS', 'Predicted_PTS'] + features
export_data = all_seasons[export_columns]

# Export to Excel
export_data.to_excel('nba_player_predictions.xlsx', index=False)

print("Data has been successfully exported to 'nba_player_predictions.xlsx'.")



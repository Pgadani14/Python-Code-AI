from nba_api.stats.endpoints import leaguedashplayerstats
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

stats_2024 = leaguedashplayerstats.LeagueDashPlayerStats(season="2023-24").get_data_frames()[0]
stats_2022 = leaguedashplayerstats.LeagueDashPlayerStats(season="2021-22").get_data_frames()[0]
stats_2023 = leaguedashplayerstats.LeagueDashPlayerStats(season="2022-23").get_data_frames()[0]
stats_2025 = leaguedashplayerstats.LeagueDashPlayerStats(season="2024-25").get_data_frames()[0]

top_scorers = stats_2024[['PLAYER_NAME', 'PTS']].sort_values(by='PTS', ascending=False).head(5)

print(top_scorers.head())

stats_2022['Season'] = '2021-22'
stats_2023['Season'] = '2022-23'
stats_2024['Season'] = '2023-24'

all_seasons = pd.concat([stats_2022, stats_2023, stats_2024], ignore_index=True)

# Drop rows with missing data
all_seasons_df = all_seasons.dropna()
all_seasons.to_excel('nba_player_data.xlsx', index=False)

# Features (X) and Target (y)
X = all_seasons_df[['REB', 'AST', 'STL', 'BLK', 'TOV', 'GP', 'MIN']]
y = all_seasons_df['PTS']

# Train-Test Split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features (important for some models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'R^2 Score: {r2}')

# Example new player stats (in the same feature order as training data)
new_player_stats = [[8, 5, 1, 0, 3, 35, 30]]  # Example: 8 rebounds, 5 assists, 1 steal, etc.

# Scale the new stats
new_player_scaled = scaler.transform(new_player_stats)

# Predict points
predicted_points = model.predict(new_player_scaled)
print(f'Predicted Points: {predicted_points[0]:.2f}')


#print(all_seasons.head())







# df_22 = stats_2022[['PLAYER_NAME', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'GP', 'MIN']]
# df_23 = stats_2023[['PLAYER_NAME', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'GP', 'MIN']]
# df_24 = stats_2024[['PLAYER_NAME', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'GP', 'MIN']]

# df_22 = df_22.dropna()
# df_23 = df_23.dropna()
# df_24 = df_24.dropna()





#top_scorers = df[['PLAYER_NAME', 'PTS']].sort_values(by='PTS', ascending=False).head(5)

#print(stats_2025.head())
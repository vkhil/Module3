# city_similarity.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

# Sample Data 
data = {
    'City': ['San Francisco', 'Vancouver', 'Seattle', 'Portland', 'Melbourne', 'Sydney', 'Toronto', 'London', 'Tokyo', 'Cape Town'],
    'Country': ['USA', 'Canada', 'USA', 'USA', 'Australia', 'Australia', 'Canada', 'UK', 'Japan', 'South Africa'],
    'Population': [884363, 675218, 724745, 653115, 4627345, 5312163, 2731571, 8982000, 13929286, 433688],
    'Area (sq km)': [121.4, 114.97, 217.0, 376.6, 9992.5, 12368.0, 630.2, 1572, 2191, 400.3],
    'Population Density': [7279, 5869, 3339, 1734, 463, 429, 4334, 5713, 6357, 1084],
    'GDP per Capita': [112372, 50644, 69413, 57638, 55100, 56587, 49600, 46400, 40400, 12340],
    'Unemployment Rate (%)': [3.2, 5.0, 4.6, 4.9, 5.4, 4.9, 7.0, 4.1, 2.8, 27.0],
    'Cost of Living Index': [91.64, 75.72, 72.80, 71.50, 67.87, 69.60, 66.69, 83.95, 82.80, 42.23],
    'Average Temperature (°C)': [15.2, 11.0, 10.7, 12.2, 14.5, 17.5, 8.4, 10.2, 16.0, 17.5],
    'Annual Precipitation (mm)': [600, 1150, 950, 882, 650, 1220, 833, 690, 1520, 515],
    'Safety Index': [45.7, 61.2, 60.1, 64.3, 60.9, 60.1, 57.0, 57.6, 69.1, 36.7]
}

# Convert the dictionary to a pandas DataFrame
df = pd.DataFrame(data)

# Select the features to use for similarity calculation
features = ['Population Density', 'GDP per Capita', 'Cost of Living Index', 'Average Temperature (°C)', 'Annual Precipitation (mm)', 'Safety Index']

# Normalize the feature values (standardization)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[features])

# Calculate Euclidean distances between cities
distance_matrix = euclidean_distances(scaled_features)

# Convert the distance matrix to a DataFrame for easier interpretation
distance_df = pd.DataFrame(distance_matrix, index=df['City'], columns=df['City'])

# Function to find the top N similar cities for a given city
def find_similar_cities(city, top_n=5):
    if city not in distance_df.columns:
        raise ValueError(f"City {city} not found in the dataset")
    
    # Get the distances to all other cities and sort them
    similar_cities = distance_df[city].sort_values()[1:top_n+1]  # Skip the first one (which is the city itself)
    return similar_cities

# Test the function with the query city
query_city = 'San Francisco'
top_similar_cities = find_similar_cities(query_city, top_n=10)

# Output the top 10 most similar cities to the query city
print(f"Top 10 cities most similar to {query_city}:")
print(top_similar_cities)

# Save the distance matrix to a CSV file for further analysis
distance_df.to_csv('city_similarity_matrix.csv', index=True)

# Save the top similar cities as a separate file
top_similar_cities.to_csv(f'top_10_similar_cities_to_{query_city}.csv', index=True)

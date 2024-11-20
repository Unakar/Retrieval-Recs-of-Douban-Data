import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score

data = pd.read_csv('data/movie_score.csv')

data['Time'] = pd.to_datetime(data['Time'])
data['Year'] = data['Time'].dt.year
data['Month'] = data['Time'].dt.month
data['Day'] = data['Time'].dt.day
data['Hour'] = data['Time'].dt.hour

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Create user-item rating matrix
user_item_matrix = train_data.pivot_table(index='User', columns='Movie', values='Rate').fillna(0).values

user_similarity = cosine_similarity(user_item_matrix)

# Map user IDs to matrix indices
user_id_to_index = {user_id: idx for idx, user_id in enumerate(train_data['User'].unique())}

# Map item IDs to matrix indices
item_id_to_index = {item_id: idx for idx, item_id in enumerate(train_data['Movie'].unique())}


def predict_ratings(user_id, item_id):
    
    user_index = user_id_to_index.get(user_id)
    if user_index is None:
        return np.nan  
    item_index = item_id_to_index.get(item_id)
    if item_index is None:
        return np.nan 
    
    user_similarities = user_similarity[user_index]
    item_ratings = user_item_matrix[:, item_index]
    
    weighted_sum = np.dot(user_similarities, item_ratings)
    sum_similarities = np.sum(user_similarities)
    predicted_rating = weighted_sum / sum_similarities if sum_similarities != 0 else np.nan
    
    return predicted_rating

def predict_ratings_with_context(user_id, item_id, year, month, day, hour):
    pass

# Function to evaluate NDCG
def evaluate_ndcg(test_data, predict_func):
    user_ids = test_data['User'].unique()
    ndcg_scores = []
    
    for user_id in user_ids:
        user_test_data = test_data[test_data['User'] == user_id]
        item_ids = user_test_data['Movie'].values
        true_ratings = user_test_data['Rate'].values
        
        predicted_ratings = []
        for item_id in item_ids:
            if predict_func == predict_ratings:
                predicted_rating = predict_func(user_id, item_id)
            else:
                timestamp = user_test_data[user_test_data['Movie'] == item_id]['Time'].values[0]
                timestamp = pd.to_datetime(timestamp)  # Convert numpy.datetime64 to datetime
                year = timestamp.year
                month = timestamp.month
                day = timestamp.day
                hour = timestamp.hour
                predicted_rating = predict_func(user_id, item_id, year, month, day, hour)
            
            # Skip the sample if predicted rating is NaN
            if np.isnan(predicted_rating):
                continue
            
            predicted_ratings.append(predicted_rating)
        
        # Skip the user if there are no valid predicted ratings
        if len(predicted_ratings) == 0:
            continue
        
        # Ensure true ratings and predicted ratings have the same length
        if len(true_ratings) != len(predicted_ratings):
            continue
        
        ndcg_scores.append(ndcg_score([true_ratings], [predicted_ratings]))
    
    return np.mean(ndcg_scores) if ndcg_scores else 0

# Evaluate NDCG for both prediction functions
ndcg_without_context = evaluate_ndcg(test_data, predict_ratings)
#ndcg_with_context = evaluate_ndcg(test_data, predict_ratings_with_context)

print(f"NDCG without context: {ndcg_without_context}")
#print(f"NDCG with context: {ndcg_with_context}")

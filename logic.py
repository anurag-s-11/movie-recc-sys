import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load Datasets
movies = pd.read_csv('tmdb_5000_movies.csv')
ratings = pd.read_csv('ratingss.csv')

# --- LAYER 1: SiMoI-Based Imputation ---
def apply_simoi_imputation(matrix):
    filled_matrix = matrix.copy()
    print("Executing SiMoI Imputation to resolve data sparsity...")
    
    for col in matrix.columns:
        if matrix[col].isnull().any():
            col_mode = matrix[col].mode()
            mode_val = col_mode[0] if not col_mode.empty else 3.0
            min_val = matrix[col].min() if not pd.isna(matrix[col].min()) else 1.0
            imputed_val = (mode_val + min_val) / 2
            filled_matrix[col] = filled_matrix[col].fillna(imputed_val)
    return filled_matrix

# Create Matrix
user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')
imputed_matrix = apply_simoi_imputation(user_item_matrix)

# Compute Similarity
print("Generating base collaborative similarity matrix...")
movie_sim_matrix = cosine_similarity(imputed_matrix.T)
movie_sim_df = pd.DataFrame(movie_sim_matrix, index=imputed_matrix.columns, columns=imputed_matrix.columns)

# --- THE FIX FOR THE ERROR ---
# Convert columns to standard 'object' type to avoid Pandas StringDtype pickle errors
movies_metadata = movies[['id', 'title', 'genres']].copy()
movies_metadata['title'] = movies_metadata['title'].astype(object)
movies_metadata['genres'] = movies_metadata['genres'].astype(object)

# 2. Save the Architecture Components
with open('facf_mrs_model.pkl', 'wb') as f:
    pickle.dump({
        'movies_metadata': movies_metadata,
        'similarity_matrix': movie_sim_df
    }, f)

print("Logic layer successful. Fixed 'facf_mrs_model.pkl' has been generated.")
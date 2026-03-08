import streamlit as st
import pandas as pd
import pickle
import requests
import ast

# Load Model
with open('facf_mrs_model.pkl', 'rb') as f:
    data = pickle.load(f)
    movies_df, movie_sim = data['movies_metadata'], data['similarity_matrix']

def fetch_poster(movie_id):
    #api_key = "76995d3c6fd91a2284b4ad8cb390c7b8"
    api_key = "KGAT_72dc0f37d8fc9014d867095b6aea9158"
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}"
    try:
        res = requests.get(url).json()
        return f"https://image.tmdb.org/t/p/w500{res.get('poster_path')}"
    except:
        return "https://via.placeholder.com/500"

def get_recommendations(movie_title):
    target = movies_df[movies_df['title'] == movie_title].iloc[0]
    target_id = target['id']
    target_genres = set([g['name'] for g in ast.literal_eval(target['genres'])])
    
    if target_id not in movie_sim.index: return []
    
    sim_scores = movie_sim[target_id].sort_values(ascending=False)
    final_recs = []
    for m_id in sim_scores.index[1:]:
        if len(final_recs) >= 10: break
        cand = movies_df[movies_df['id'] == m_id]
        if not cand.empty:
            cand_genres = set([g['name'] for g in ast.literal_eval(cand.iloc[0]['genres'])])
            if target_genres.intersection(cand_genres): # Our Architectural Twist
                final_recs.append((m_id, cand.iloc[0]['title']))
    return final_recs

st.title("FACF Movie Recommender")
selected = st.selectbox("Select Movie:", movies_df['title'].values)

if st.button("Recommend"):
    recs = get_recommendations(selected)
    cols = st.columns(5)
    for i in range(10):
        if i < len(recs):
            with cols[i % 5]:
                st.image(fetch_poster(recs[i][0]))
                st.caption(recs[i][1])
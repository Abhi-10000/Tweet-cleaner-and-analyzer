# app.py
import streamlit as st
import sqlite3
import pandas as pd
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from pprint import pprint
from agent import analyze_text_with_agent
import numpy as np

# --- Our Custom Theme Map (Unchanged) ---
CLUSTER_NAME_MAP = {
    0: 'Personal Conversations & Banter', 1: 'Weather & Seasons', 2: 'Friends, Fun & Daily Life', 
    3: 'Short Replies & Random Thoughts', 4: 'Sickness, Pain & Health', 5: 'Thanks, Follows & Appreciation', 
    6: 'Fandom, Media & Eclectic Chatter', 7: 'Twitter Platform & Tweeting', 8: 'Personal Updates & Replies', 
    9: 'Specific Days, Events, & Plans', 10: 'Technology, Software, & Mobile Devices', 11: 'Locations, Travel, & Events', 
    12: 'Music & Media Entertainment', 13: 'Missing People, Places, & Sadness', 14: 'Personal Conversations & Social Updates', 
    15: 'Food, Drinks, & Eating', 16: 'Sleep, Bedtime, & Tiredness', 17: 'Watching TV, Movies, & Shows', 
    18: 'Discussing Work & Jobs', 19: 'School, Exams, & Student Life'
}
# --------------------------------------

# --- Model & Data Loading (with Caching) ---

# @st.cache_resource is a Streamlit decorator.
# It tells Streamlit to run this function once, store the result in memory,
# and reuse it for all subsequent users/sessions. This is critical
# so we don't reload the models (which are slow) every time.
@st.cache_resource
def load_models():
    """Loads all models and data needed for the app."""
    print("Loading models into cache...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    theme_embeddings = np.load('cluster_centroids.npy')
    theme_ids = list(CLUSTER_NAME_MAP.keys())
    print("Models loaded successfully.")
    return model, theme_embeddings, theme_ids

# --- Core Functions (from main.py) ---

def find_best_matching_theme(query, theme_ids, theme_embeddings, model):
    """Finds the most semantically similar theme to the user's query."""
    query_embedding = model.encode(query)
    similarities = [1 - cosine(query_embedding, theme_emb) for theme_emb in theme_embeddings]
    best_match_index = np.argmax(similarities)
    return theme_ids[best_match_index]

def query_database_by_theme(theme_id, db_name='tweet_analysis.db'):
    """Queries the database to get all tweets for a specific theme ID."""
    conn = sqlite3.connect(db_name)
    query = f"SELECT * FROM tweets WHERE theme = ? ORDER BY likes DESC"
    df = pd.read_sql_query(query, conn, params=(int(theme_id),))
    conn.close()
    return df

# ---  streamlit UI ---

# Set the page title and icon
st.set_page_config(page_title="Hybrid-Intelligence Analysis", page_icon="📈")

# Load our models once at the start
embedding_model, theme_embeddings, theme_ids = load_models()

# --- Main App Interface ---
st.title("📈 Hybrid-Intelligence Social Media Analysis")
st.write("Enter a topic or a sample tweet to analyze. The system will find the most relevant theme in the database, run a full quantitative analysis, and then use an LLM to provide a final qualitative summary.")

# Create a form for the user input
with st.form(key="analysis_form"):
    user_query = st.text_input("Enter your query:", "I am feeling sick today")
    submit_button = st.form_submit_button(label="Analyze Topic 📊")

# --- Analysis Logic ---
if submit_button and user_query:
    with st.spinner("Analyzing... This may take a moment."):
        
        # --- Step 1: Semantic Search ---
        matched_theme_id = find_best_matching_theme(user_query, theme_ids, theme_embeddings, embedding_model)
        matched_theme_name = CLUSTER_NAME_MAP[matched_theme_id]
        
        # --- Step 2: Data Retrieval ---
        results_df = query_database_by_theme(matched_theme_id)
        
        if results_df.empty:
            st.error("No data found for this theme in the database.")
        else:
            # --- Step 3: Advanced Quantitative Aggregation ---
            total_tweets = len(results_df)
            sentiment_counts = results_df['sentiment'].value_counts()
            
            # Ensure both 0 (Neg) and 1 (Pos) keys exist
            pos_count = sentiment_counts.get(1, 0)
            neg_count = sentiment_counts.get(0, 0)
            
            # Engagement Metrics
            positive_tweets_df = results_df[results_df['sentiment'] == 1]
            negative_tweets_df = results_df[results_df['sentiment'] == 0]
            
            total_positive_engagement = positive_tweets_df['likes'].sum() + positive_tweets_df['retweets'].sum()
            total_negative_engagement = negative_tweets_df['likes'].sum() + negative_tweets_df['retweets'].sum()
            
            impact_score = total_positive_engagement - (total_negative_engagement * 0.5)

            # --- Step 4: LLM Synthesis ---
            top_tweets_text = "\n".join(f"- {text}" for text in results_df['raw_text'].head(10))
            
            llm_prompt = (
                f"User Query: {user_query}\n"
                f"Matched Theme: {matched_theme_name}\n"
                f"Total Tweets Found: {total_tweets}\n"
                f"Sentiment Distribution (by count): Positive: {pos_count}, Negative: {neg_count}\n"
                f"Total Positive Engagement (Likes+Retweets): {total_positive_engagement}\n"
                f"Total Negative Engagement (Likes+Retweets): {total_negative_engagement}\n\n"
                f"Top 10 Most-Liked Tweets (for context):\n{top_tweets_text}"
            )
            final_summary = analyze_text_with_agent(llm_prompt)

            # --- Step 5: Display Final Report ---
            st.success(f"Query matched to theme: **{matched_theme_name}**")

            st.subheader("📈 Quantitative Summary")
            
            # Display key metrics in columns
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Tweets", f"{total_tweets:,.0f}")
            col2.metric("Impact Score", f"{impact_score:,.0f}")
            
            st.text("Sentiment Distribution (by Tweet Count):")
            st.write(f"Positive: {pos_count} | Negative: {neg_count}")
            
            # Calculate percentages for the progress bar
            if total_tweets > 0:
                pos_percent = pos_count / total_tweets
            st.progress(pos_percent, text=f"{pos_percent*100:.1f}% Positive")
            
            st.text("Sentiment Distribution (by Engagement):")
            st.write(f"Positive Engagement: {total_positive_engagement:,.0f}")
            st.write(f"Negative Engagement: {total_negative_engagement:,.0f}")
            
            
            st.subheader("🤖 Qualitative Summary from LLM Agent")
            if final_summary:
                st.json(final_summary)
            else:
                st.error("Could not generate LLM summary.")
                
            # Use an expander to hide the "context" tweets
            with st.expander("View Top 10 Most-Liked Tweets (Context for LLM)"):
                st.dataframe(results_df[['raw_text', 'likes', 'retweets']].head(10))
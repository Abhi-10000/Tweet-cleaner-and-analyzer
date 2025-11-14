# main.py
import sqlite3
import pandas as pd
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from collections import Counter
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

def get_theme_embeddings():
    """Generates vector embeddings for our predefined list of themes."""
    print("Loading embedding model and cluster centroids...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    theme_embeddings = np.load('cluster_centroids.npy')
    theme_ids = list(CLUSTER_NAME_MAP.keys())
    print("Theme embeddings and IDs loaded.")
    return theme_ids, theme_embeddings, model

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

if __name__ == '__main__':
    theme_ids, theme_embeddings, embedding_model = get_theme_embeddings()

    while True:
        print("\n" + "="*50)
        user_query = input("Enter a topic to analyze (or type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break

        # --- Step 1: Semantic Search (Unchanged) ---
        matched_theme_id = find_best_matching_theme(user_query, theme_ids, theme_embeddings, embedding_model)
        matched_theme_name = CLUSTER_NAME_MAP[matched_theme_id]
        print(f"\nUser query '{user_query}' matched to theme: '{matched_theme_name}' (Cluster ID: {matched_theme_id})")

        # --- Step 2: Data Retrieval (Unchanged) ---
        print(f"Retrieving pre-analyzed data for '{matched_theme_name}' from the database...")
        results_df = query_database_by_theme(matched_theme_id)
        
        if results_df.empty:
            print("No data found for this theme in the database.")
            continue

        # --- Step 3: Advanced Quantitative Aggregation (NEW) ---
        
        # 3a. Simple Tweet Counts
        total_tweets = len(results_df)
        sentiment_counts = results_df['sentiment'].value_counts()
        sentiment_dist = {('Negative' if k == 0 else 'Positive'): v for k, v in sentiment_counts.items()}
        
        # 3b. Advanced Engagement Metrics
        # (sentiment is 0 for Negative, 1 for Positive)
        positive_tweets_df = results_df[results_df['sentiment'] == 1]
        negative_tweets_df = results_df[results_df['sentiment'] == 0]
        
        total_positive_likes = positive_tweets_df['likes'].sum()
        total_positive_retweets = positive_tweets_df['retweets'].sum()
        total_positive_engagement = total_positive_likes + total_positive_retweets
        
        total_negative_likes = negative_tweets_df['likes'].sum()
        total_negative_retweets = negative_tweets_df['retweets'].sum()
        total_negative_engagement = total_negative_likes + total_negative_retweets
        
        total_engagement = total_positive_engagement + total_negative_engagement
        
        # 3c. Custom "Impact Score"
        # Let's define it as: (Total Positive Engagement) - (Total Negative Engagement * 0.5)
        # This rewards positive engagement but only lightly penalizes negative engagement
        impact_score = total_positive_engagement - (total_negative_engagement * 0.5)

        # --- Step 4: LLM Synthesis (Unchanged) ---
        top_tweets_text = "\n".join(f"- {text}" for text in results_df['raw_text'].head(10))
        
        # We can now pass the LLM an even better prompt
        llm_prompt = (
            f"User Query: {user_query}\n"
            f"Matched Theme: {matched_theme_name}\n"
            f"Total Tweets Found: {total_tweets}\n"
            f"Sentiment Distribution (by count): {sentiment_dist}\n"
            f"Total Positive Engagement (Likes+Retweets): {total_positive_engagement}\n"
            f"Total Negative Engagement (Likes+Retweets): {total_negative_engagement}\n\n"
            f"Top 10 Most-Liked Tweets (for context):\n{top_tweets_text}"
        )
        final_summary = analyze_text_with_agent(llm_prompt)

        # --- Step 5: Display Final Report (NEW) ---
        print("\n--- 📈 Final Analysis Report 📊 ---")
        print(f"Analysis for topic related to: '{matched_theme_name}'")
        print(f"Based on {total_tweets} pre-analyzed tweets.")
        
        print("\nQuantitative Summary (by Tweet Count):")
        for sentiment, count in sentiment_dist.items():
            percentage = (count / total_tweets) * 100
            print(f"   - {sentiment}: {count} tweets ({percentage:.2f}%)")
        
        print("\nQuantitative Summary (by Engagement):")
        print(f"   - Total Positive Engagement (Likes+RTs): {total_positive_engagement:,.0f}")
        print(f"   - Total Negative Engagement (Likes+RTs): {total_negative_engagement:,.0f}")
        print(f"   - Overall Topic Impact Score: {impact_score:,.0f}")

        
        print("\nQualitative Summary from LLM Agent:")
        if final_summary:
            pprint(final_summary)
        else:
            print("Could not generate LLM summary.")
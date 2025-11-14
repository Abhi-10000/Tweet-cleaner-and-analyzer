# enrich_database.py
import sqlite3
import pandas as pd
import joblib
import time
from text_preprocessor_tool import preprocess_text
from sentence_transformers import SentenceTransformer
import numpy as np # <-- NEW: For generating random data

def setup_database(db_name='tweet_analysis.db'):
    """Creates the SQLite database and the tweets table if they don't exist."""
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    # --- NEW SCHEMA ---
    # We are adding 'likes' and 'retweets' columns
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tweets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            raw_text TEXT,
            cleaned_text TEXT,
            sentiment INTEGER,
            theme INTEGER,
            likes INTEGER,
            retweets INTEGER
        )
    ''')
    conn.commit()
    conn.close()
    print(f"Database '{db_name}' is set up with new schema (including likes/retweets).")

if __name__ == '__main__':
    start_time = time.time()
    DB_NAME = 'tweet_analysis.db'
    
    # 1. Setup the database
    setup_database(DB_NAME)

    # 2. Load all trained assets
    print("Loading trained models and vectorizers...")
    sentiment_model = joblib.load('sentiment_model.pkl')
    sentiment_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    kmeans_model = joblib.load('kmeans_model.pkl')
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2') 
    print("Assets loaded.")

    # 3. Load the "new" raw data to be processed
    print("Loading raw data to process...")
    DATASET_PATH = 'sentiment140.csv'
    column_names = ['target', 'ids', 'date', 'flag', 'user', 'text']
    df = pd.read_csv(DATASET_PATH, encoding='latin-1', names=column_names)
    
    df_sample = df.sample(n=20000, random_state=42)
    print(f"Processing {len(df_sample)} new tweets...")

    # 4. Connect to the database
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # 5. Process and insert data in a loop
    for index, row in df_sample.iterrows():
        raw_tweet = row['text']
        cleaned_tweet = preprocess_text(raw_tweet)
        
        if not cleaned_tweet:
            continue
            
        # --- Sentiment Prediction (Unchanged) ---
        sentiment_vector = sentiment_vectorizer.transform([cleaned_tweet])
        sentiment = sentiment_model.predict(sentiment_vector)[0]
        
        # --- Theme Prediction (Unchanged) ---
        tweet_embedding = embedding_model.encode([cleaned_tweet])
        cluster_id = kmeans_model.predict(tweet_embedding)[0]
        
        # --- NEW: Simulate Engagement Data ---
        likes = np.random.randint(0, 5000)
        retweets = np.random.randint(0, 1000)
        
        # --- NEW: Updated INSERT statement ---
        cursor.execute('''
            INSERT INTO tweets (raw_text, cleaned_text, sentiment, theme, likes, retweets)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (raw_tweet, cleaned_tweet, int(sentiment), int(cluster_id), likes, retweets))

        current_row_num = index + 1
        if current_row_num % 1000 == 0:
            print(f"   Processed {current_row_num}/{len(df_sample)} tweets...")
    
    # Commit changes and close the connection
    conn.commit()
    conn.close()
    
    end_time = time.time()
    print("\nDatabase enrichment complete with new cluster IDs and engagement data.")
    print(f"Total time taken: {end_time - start_time:.2f} seconds.")
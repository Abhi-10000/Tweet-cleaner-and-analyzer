# test_single_tweet.py
import joblib
import sqlite3
from text_preprocessor_tool import preprocess_text
from sklearn.datasets import fetch_20newsgroups

def classify_and_store_tweet(raw_tweet, models):
    """
    Analyzes a single new tweet using traditional ML models and stores it in the DB.
    
    Args:
        raw_tweet (str): The raw text of the tweet.
        models (dict): A dictionary containing the loaded models and vectorizers.
    """
    print("\n" + "-"*50)
    print(f"Original Tweet: \n{raw_tweet}")
    
    # 1. Preprocess the text
    cleaned_tweet = preprocess_text(raw_tweet)
    print(f"\nCleaned Tweet: \n{cleaned_tweet}")

    if not cleaned_tweet:
        print("\nTweet is empty after cleaning. Cannot classify.")
        return

    # 2. Predict Sentiment
    sentiment_vector = models['sentiment_vectorizer'].transform([cleaned_tweet])
    sentiment_id = models['sentiment_model'].predict(sentiment_vector)[0]
    sentiment = "Positive" if sentiment_id == 1 else "Negative"
    
    # 3. Predict Theme
    theme_vector = models['theme_vectorizer'].transform([cleaned_tweet])
    theme_id = models['theme_model'].predict(theme_vector)[0]
    theme = models['theme_names'][theme_id]

    print("\n--- 🧠 ML Model Analysis ---")
    print(f"Predicted Sentiment: {sentiment}")
    print(f"Predicted Theme: {theme}")
    
    # 4. Store in the database
    try:
        conn = sqlite3.connect('tweet_analysis.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO tweets (raw_text, cleaned_text, sentiment, theme)
            VALUES (?, ?, ?, ?)
        ''', (raw_tweet, cleaned_tweet, sentiment_id, theme))
        conn.commit()
        conn.close()
        print("\n✅ Successfully saved to the database.")
    except Exception as e:
        print(f"\n❌ Error saving to database: {e}")

if __name__ == '__main__':
    # Load all models and assets once at the start
    print("Loading ML models... This may take a moment.")
    loaded_models = {
        "sentiment_model": joblib.load('sentiment_model.pkl'),
        "sentiment_vectorizer": joblib.load('tfidf_vectorizer.pkl'),
        "theme_model": joblib.load('theme_model.pkl'),
        "theme_vectorizer": joblib.load('theme_vectorizer.pkl'),
        "theme_names": fetch_20newsgroups(subset='all').target_names
    }
    print("Models loaded successfully.")
    
    # Main loop to take user input
    while True:
        new_tweet = input("\nEnter a new tweet to analyze (or type 'exit' to quit): ")
        if new_tweet.lower() == 'exit':
            break
        
        classify_and_store_tweet(new_tweet, loaded_models)
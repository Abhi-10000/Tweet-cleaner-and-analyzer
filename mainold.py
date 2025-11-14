# main.py
import pandas as pd
from data_loader_tool import load_and_filter_data
from text_preprocessor_tool import preprocess_text
from agent import analyze_text_with_agent
from collections import Counter

# --- Configuration ---
DATASET_PATH = 'sentiment140.csv'
SEARCH_KEYWORD = 'love'
SAMPLE_SIZE = 50 # Let's analyze a bigger sample to get meaningful stats

print(f"--- Starting Full Tweet Analysis Pipeline for '{SEARCH_KEYWORD}' ---")

# --- Step 1: Load and Filter Data ---
tweets_df = load_and_filter_data(DATASET_PATH, SEARCH_KEYWORD, sample_size=SAMPLE_SIZE)

if tweets_df is not None and not tweets_df.empty:
    
    results = []
    print(f"\n--- Analyzing {len(tweets_df)} tweets... ---")

    # --- Step 2 & 3: Loop, Preprocess, and Analyze ---
    for index, row in tweets_df.iterrows():
        original_tweet = row['text']
        
        # Clean the tweet
        cleaned_text = preprocess_text(original_tweet)
        
        # Skip empty tweets after cleaning
        if not cleaned_text:
            continue
            
        # Analyze with the agent
        analysis = analyze_text_with_agent(cleaned_text)
        
        if analysis:
            results.append(analysis)
            print(f"  Analyzed tweet {index+1}/{len(tweets_df)}") # Progress indicator

    # --- Step 4: Aggregate Results and Generate Final Report ---
    if results:
        print("\n--- Final Analysis Report ---")
        
        # Calculate sentiment distribution
        sentiment_counts = Counter(res['sentiment'] for res in results)
        total_results = len(results)
        
        print("\n📊 Overall Sentiment Distribution:")
        for sentiment, count in sentiment_counts.items():
            percentage = (count / total_results) * 100
            print(f"- {sentiment}: {count} tweets ({percentage:.2f}%)")
            
        # Calculate top themes
        theme_counts = Counter(res['theme'] for res in results)
        
        print("\n🔑 Top Themes Discussed:")
        for theme, count in theme_counts.most_common(5): # Show top 5 themes
            print(f"- {theme}: mentioned {count} times")

        # Show a few examples
        print("\n📝 Example Analyses:")
        for i, res in enumerate(results[:3]): # Show first 3 results
            print(f"\nExample {i+1}:")
            print(f"  Sentiment: {res['sentiment']}")
            print(f"  Theme: {res['theme']}")
            print(f"  Summary: {res['summary']}")

print("\n--- Pipeline Run Complete ---")
# inspect_clusters.py
import sqlite3
import pandas as pd

DB_NAME = 'tweet_analysis.db'
NUM_CLUSTERS = 20  # The number of clusters you trained for
SAMPLE_SIZE = 15   # Number of tweets to show per cluster

def get_cluster_samples(db_name, cluster_id, sample_size):
    """Fetches a random sample of tweets for a given cluster ID."""
    conn = sqlite3.connect(db_name)
    query = f"SELECT raw_text FROM tweets WHERE theme = ? ORDER BY RANDOM() LIMIT ?"
    df = pd.read_sql_query(query, conn, params=(cluster_id, sample_size))
    conn.close()
    return df['raw_text'].tolist()

if __name__ == '__main__':
    print("--- Inspecting Clusters to Create Names ---")
    
    # We will build our theme map here
    cluster_name_map = {}

    for i in range(NUM_CLUSTERS):
        print("\n" + "="*50)
        print(f"Cluster ID: {i}")
        print("="*50)
        
        samples = get_cluster_samples(DB_NAME, i, SAMPLE_SIZE)
        
        if not samples:
            print("No tweets found for this cluster.")
            continue
            
        for j, tweet in enumerate(samples):
            print(f"  {j+1:02d}: {tweet}")
        
        # This is the interactive part
        print("-" * 50)
        theme_name = input(f"What is a good name for Cluster {i}? (or 'skip') ")
        
        if theme_name.lower() != 'skip':
            cluster_name_map[i] = theme_name

    print("\n\n--- 🚀 Your Final Cluster Name Map ---")
    print("You can copy this dictionary into main.py")
    print(cluster_name_map)
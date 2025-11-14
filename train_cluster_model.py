# train_cluster_model.py
import pandas as pd
import joblib
import time
from text_preprocessor_tool import preprocess_text
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np

print("--- Starting Unsupervised Theme Model Training ---")
start_time = time.time()

# 1. Load Data
print("Loading raw data...")
DATASET_PATH = 'sentiment140.csv'
column_names = ['target', 'ids', 'date', 'flag', 'user', 'text']
df = pd.read_csv(DATASET_PATH, encoding='latin-1', names=column_names)

# We'll use a 20k sample to train the clustering model
df_sample = df.sample(n=20000, random_state=42)

# 2. Preprocess Text
print(f"Preprocessing {len(df_sample)} tweets...")
df_sample['cleaned_text'] = df_sample['text'].apply(preprocess_text)
# Drop any rows that became empty after cleaning
processed_texts = df_sample['cleaned_text'].dropna()
processed_texts = processed_texts[processed_texts != ""]
print(f"Successfully preprocessed {len(processed_texts)} tweets.")

# 3. Create Vector Embeddings
print("Loading sentence embedding model...")
# This model creates 384-dimensional vectors
model = SentenceTransformer('all-MiniLM-L6-v2') 

print("Creating embeddings... (This may take a few minutes)")
corpus_embeddings = model.encode(processed_texts.tolist(), show_progress_bar=True)

# 4. Train K-Means Clustering Model
# Let's aim for 20 clusters, similar to the 20 Newsgroups
num_clusters = 20
print(f"Training K-Means model to find {num_clusters} clusters...")
kmeans_model = KMeans(n_clusters=num_clusters, init='k-means++', n_init=10, random_state=42)
kmeans_model.fit(corpus_embeddings)

# 5. Save the Model and Cluster Centroids
print("Training complete. Saving assets...")

# Save the main K-Means model
joblib.dump(kmeans_model, 'kmeans_model.pkl')

# Save the cluster centroids. These are the "average" vectors for each cluster.
# Our main.py will use these for semantic search.
centroids = kmeans_model.cluster_centers_
np.save('cluster_centroids.npy', centroids)

end_time = time.time()
print(f"--- Successfully trained and saved 'kmeans_model.pkl' and 'cluster_centroids.npy' ---")
print(f"Total time taken: {end_time - start_time:.2f} seconds.")

# 6. (Optional) Inspect the clusters to name them
print("\n--- Top 10 Tweets from Cluster 0 (for inspection) ---")
# Get all tweets assigned to cluster 0
cluster_0_indices = np.where(kmeans_model.labels_ == 0)[0]
# Get the original text for those indices
original_texts = df_sample.iloc[cluster_0_indices]['text']
print(original_texts.head(10))
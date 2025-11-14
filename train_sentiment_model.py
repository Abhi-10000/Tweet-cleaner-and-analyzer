# train_sentiment_model.py
import pandas as pd
import re
import time
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib # For saving the model and vectorizer

# --- Text Preprocessing Function (same as before) ---
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    return ' '.join(lemmatized_tokens)

# --- Main Training Logic ---
if __name__ == '__main__':
    start_time = time.time()
    
    # 1. Load a sample of the data to keep training time manageable
    print("Loading data...")
    DATASET_PATH = 'sentiment140.csv'
    column_names = ['target', 'ids', 'date', 'flag', 'user', 'text']
    df = pd.read_csv(DATASET_PATH, encoding='latin-1', names=column_names)
    
    # Take a balanced sample to avoid long training times
    sample_size_per_class = 50000 # 50k positive, 50k negative
    df_positive = df[df['target'] == 4].sample(n=sample_size_per_class, random_state=42)
    df_negative = df[df['target'] == 0].sample(n=sample_size_per_class, random_state=42)
    df_sample = pd.concat([df_positive, df_negative])

    # Map target labels: 0 stays 0 (Negative), 4 becomes 1 (Positive)
    df_sample['target'] = df_sample['target'].map({0: 0, 4: 1})
    print(f"Data loaded and sampled. Total rows: {len(df_sample)}")

    # 2. Preprocess the text data
    print("Preprocessing text data...")
    df_sample['cleaned_text'] = df_sample['text'].apply(preprocess_text)
    
    # 3. Split data into training and testing sets
    X = df_sample['cleaned_text']
    y = df_sample['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Vectorize the text using TF-IDF
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=10000) # Limit to top 10,000 features
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test) # IMPORTANT: Only transform the test set
    
    # 5. Train a Logistic Regression model
    print("Training the sentiment model...")
    model = LogisticRegression(solver='liblinear')
    model.fit(X_train_tfidf, y_train)
    
    # 6. Evaluate the model
    print("Evaluating the model...")
    y_pred = model.predict(X_test_tfidf)
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    
    # 7. Save the trained model and the vectorizer
    print("Saving the model and vectorizer...")
    joblib.dump(model, 'sentiment_model.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    
    end_time = time.time()
    print(f"\nTraining complete. Files 'sentiment_model.pkl' and 'tfidf_vectorizer.pkl' have been saved.")
    print(f"Total time taken: {end_time - start_time:.2f} seconds.")
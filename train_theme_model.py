# train_theme_model.py
import time
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import joblib

# --- Main Training Logic ---
if __name__ == '__main__':
    start_time = time.time()

    # 1. Load the 20 Newsgroups dataset
    # We remove headers, footers, and quotes to ensure the model learns from the actual content.
    print("Loading the 20 Newsgroups dataset...")
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    
    # The data contains the text, and the target contains the category number for each text.
    X = newsgroups.data
    y = newsgroups.target
    target_names = newsgroups.target_names # Human-readable names of the categories
    print("Dataset loaded successfully.")

    # 2. Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. Vectorize the text using TF-IDF
    # We don't need a separate cleaning function as fetch_20newsgroups is already quite clean.
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=15000, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # 4. Train a Logistic Regression model for multi-class classification
    print("Training the thematic model...")
    # 'ovr' (One-vs-Rest) is a common strategy for multi-class problems.
    model = LogisticRegression(solver='liblinear', multi_class='ovr')
    model.fit(X_train_tfidf, y_train)

    # 5. Evaluate the model
    print("Evaluating the model...")
    y_pred = model.predict(X_test_tfidf)
    print("\n--- Classification Report ---")
    # This report will show the performance for each of the 20 topics.
    print(classification_report(y_test, y_pred, target_names=target_names))

    # 6. Save the trained model and the vectorizer
    print("Saving the model and vectorizer...")
    joblib.dump(model, 'theme_model.pkl')
    joblib.dump(vectorizer, 'theme_vectorizer.pkl') # Use a different name to not overwrite the sentiment one

    end_time = time.time()
    print(f"\nTraining complete. Files 'theme_model.pkl' and 'theme_vectorizer.pkl' have been saved.")
    print(f"Total time taken: {end_time - start_time:.2f} seconds.")
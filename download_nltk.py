# download_nltk.py
import nltk

print("Downloading 'punkt' for tokenization...")
nltk.download('punkt')

print("Downloading 'stopwords' for stop word removal...")
nltk.download('stopwords')

print("Downloading 'wordnet' for lemmatization...")
nltk.download('wordnet')

print("All necessary NLTK data downloaded.")
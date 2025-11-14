# text_preprocessor_tool.py
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def preprocess_text(text):
    """
    Cleans and preprocesses a single string of text.

    Args:
        text (str): The raw tweet text.

    Returns:
        str: The cleaned and processed text.
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercasing
    text = text.lower()
    
    # 2. Removing Noise (URLs, mentions, hashtags, special characters)
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove user mentions (@username)
    text = re.sub(r'@\w+', '', text)
    # Remove hashtags (#)
    text = re.sub(r'#', '', text)
    # Remove special characters and punctuation, keeping only letters and spaces
    text = re.sub(r'[^a-z\s]', '', text)
    
    # 3. Tokenization
    # Splits the text into a list of words
    tokens = word_tokenize(text)
    
    # 4. Removing Stop Words
    # These are common words that don't add much meaning (e.g., 'a', 'the', 'is')
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    # 5. Lemmatization
    # Reduces words to their base or dictionary form (e.g., 'running' -> 'run')
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    
    # 6. Join back to a single string
    cleaned_text = ' '.join(lemmatized_tokens)
    
    return cleaned_text

# Example of how to use the function
if __name__ == '__main__':
    raw_tweet = "@Foxyma2k9 I know! Its super duper old! sadness! No camera?! <-Do they still make phones w/ no camera?! But I LOVE Tmobile! LOL https://t.co/xyz"
    
    print("--- Text Preprocessing Example ---")
    print(f"Original Tweet: \n{raw_tweet}\n")
    
    cleaned_tweet = preprocess_text(raw_tweet)
    
    print(f"Cleaned Tweet: \n{cleaned_tweet}")
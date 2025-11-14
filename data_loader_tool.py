import pandas as pd

def load_and_filter_data(filepath, keyword, sample_size=1000):
    """
    Loads the Sentiment140 dataset, filters it by a keyword, and returns a sample.

    Args:
        filepath (str): The path to the sentiment140.csv file.
        keyword (str): The keyword to search for in the tweets.
        sample_size (int): The number of tweets to return for the analysis.

    Returns:
        pandas.DataFrame: A DataFrame containing the filtered and sampled tweets.
    """
    print(f"Loading data from {filepath}...")
    
    # Define the column names since the CSV file doesn't have a header
    column_names = ['target', 'ids', 'date', 'flag', 'user', 'text']
    
    try:
        # Read the CSV file into a pandas DataFrame
        # We specify the encoding because the data contains non-standard characters
        df = pd.read_csv(filepath, encoding='latin-1', names=column_names)
        
        print("Data loaded successfully.")
        
        # Filter the DataFrame to find tweets containing the keyword (case-insensitive)
        print(f"Filtering for tweets containing the keyword: '{keyword}'...")
        filtered_df = df[df['text'].str.contains(keyword, case=False, na=False)]
        
        if filtered_df.empty:
            print("No tweets found with that keyword.")
            return pd.DataFrame() # Return an empty DataFrame
        
        # Take a random sample to keep the analysis quick and manageable
        # If there are fewer tweets than sample_size, it will return all of them
        if len(filtered_df) > sample_size:
            sampled_df = filtered_df.sample(n=sample_size, random_state=42)
            print(f"Found {len(filtered_df)} tweets. Returning a random sample of {sample_size}.")
        else:
            sampled_df = filtered_df
            print(f"Found {len(filtered_df)} tweets. Returning all of them.")
            
        return sampled_df

    except FileNotFoundError:
        print(f"Error: The file was not found at {filepath}")
        return None

# This block allows us to test the function directly
if __name__ == '__main__':
    # Define the path to your dataset and a keyword to test
    DATASET_PATH = 'sentiment140.csv'
    SEARCH_KEYWORD = 'love'
    
    # Call the function
    tweets_df = load_and_filter_data(DATASET_PATH, SEARCH_KEYWORD)
    
    # If the function returned data, print the first 5 tweets
    if tweets_df is not None and not tweets_df.empty:
        print("\n--- First 5 tweets found ---")
        for index, row in tweets_df.head().iterrows():
            print(f"- {row['text']}\n")
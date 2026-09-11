import pandas as pd
import os
from tqdm import tqdm

def load_and_preprocess(raw_data_path, brand_name, output_path):
    print(f"Loading {raw_data_path}...")
    df = pd.read_csv(raw_data_path)
    
    print(f"Filtering for {brand_name} interactions...")
    # Find all tweets by the brand
    brand_tweets = df[df['author_id'] == brand_name]
    
    # Find all customer tweets that the brand replied to
    # We need to trace the conversation. 
    # A simple approach: grab all tweets where author_id == brand OR in_response_to_tweet_id is in brand_tweets['tweet_id']
    # Or, the brand replies to a tweet_id.
    # Pandas often converts integer columns with NaNs to floats. We must remove the '.0' when casting to string.
    brand_reply_to_ids = brand_tweets['in_response_to_tweet_id'].dropna().astype(str).str.replace(r'\.0$', '', regex=True).tolist()
    
    # Let's get the customer tweets
    df['tweet_id_str'] = df['tweet_id'].astype(str)
    customer_tweets = df[df['tweet_id_str'].isin(brand_reply_to_ids)]
    
    print(f"Found {len(brand_tweets)} brand tweets and {len(customer_tweets)} customer tweets that initiated a response.")
    
    # Reconstruct pairs: Customer Message -> Brand Response
    # We will merge on customer_tweets.tweet_id == brand_tweets.in_response_to_tweet_id
    
    customer_tweets_subset = customer_tweets[['tweet_id_str', 'author_id', 'text', 'created_at']].rename(
        columns={'tweet_id_str': 'customer_tweet_id', 'author_id': 'customer_id', 'text': 'customer_text', 'created_at': 'customer_created_at'}
    )
    
    brand_tweets_subset = brand_tweets[['tweet_id', 'in_response_to_tweet_id', 'text', 'created_at']].rename(
        columns={'tweet_id': 'agent_tweet_id', 'text': 'agent_text', 'created_at': 'agent_created_at'}
    )
    
    # Convert to string for safe merging, stripping '.0'
    brand_tweets_subset['in_response_to_tweet_id'] = brand_tweets_subset['in_response_to_tweet_id'].astype(str).str.replace(r'\.0$', '', regex=True)
    
    print("Merging into conversations...")
    conversations = pd.merge(
        customer_tweets_subset,
        brand_tweets_subset,
        left_on='customer_tweet_id',
        right_on='in_response_to_tweet_id',
        how='inner'
    )
    
    print(f"Reconstructed {len(conversations)} conversation pairs.")
    
    # Basic cleaning
    conversations = conversations.dropna(subset=['customer_text', 'agent_text'])
    conversations = conversations.drop_duplicates(subset=['customer_tweet_id']) # one response per customer tweet for simplicity
    
    print(f"After deduplication: {len(conversations)} unique customer interactions.")
    
    conversations.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    if not os.path.exists('data'):
        os.makedirs('data')
    load_and_preprocess("twcs.csv", "AppleSupport", "data/apple_support_conversations.csv")

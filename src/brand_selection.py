import pandas as pd
import os
import sys

def analyze_dataset(file_path):
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    
    print(f"Total tweets: {len(df)}")
    
    # Identify brands (author_ids that are not inbound when they reply, usually strings, not just digits)
    brands = df[df['inbound'] == False]['author_id'].value_counts()
    print("\nTop 10 brands by agent response count:")
    print(brands.head(10))
    
    # We want a brand with a good number of customer inquiries and agent responses.
    # Let's look closer at the top 3 brands to find the best fit.
    # We'll calculate the number of unique conversations (using in_response_to_tweet_id)
    
    top_3 = brands.head(3).index.tolist()
    
    for brand in top_3:
        print(f"\nAnalyzing brand: {brand}")
        brand_tweets = df[df['author_id'] == brand]
        print(f"Agent tweets from {brand}: {len(brand_tweets)}")
        
        # Customer tweets directed to this brand
        # A customer tweet is inbound=True and its tweet_id is in the brand's in_response_to_tweet_id
        brand_responses = df[df['author_id'] == brand].dropna(subset=['in_response_to_tweet_id'])
        customer_tweet_ids = brand_responses['in_response_to_tweet_id'].astype(str).str.split(',').explode()
        customer_tweets = df[df['tweet_id'].astype(str).isin(customer_tweet_ids)]
        print(f"Customer tweets resolved/replied to by {brand}: {len(customer_tweets)}")
        
if __name__ == "__main__":
    analyze_dataset("twcs.csv")

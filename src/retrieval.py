import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

class SimpleRetrievalBaseline:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
        self.corpus_vectors = None
        self.df = None
        
    def build_index(self, df):
        print(f"Building TF-IDF index for {len(df)} documents...")
        self.df = df.reset_index(drop=True)
        self.corpus_vectors = self.vectorizer.fit_transform(self.df['customer_text'])
        
    def retrieve_reply(self, query):
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.corpus_vectors).flatten()
        best_idx = similarities.argmax()
        
        best_match = self.df.iloc[best_idx]
        return {
            'retrieved_customer_text': best_match['customer_text'],
            'agent_reply': best_match['agent_text'],
            'similarity_score': similarities[best_idx],
            'case_id': best_match['customer_tweet_id']
        }
        
    def save(self, path='results/simple_retrieval.pkl'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'vectorizer': self.vectorizer, 'corpus_vectors': self.corpus_vectors, 'df': self.df}, f)

if __name__ == "__main__":
    print("Loading corpus...")
    df = pd.read_csv('data/apple_support_conversations.csv', low_memory=False)
    golden_ids = pd.read_csv('data/golden_set.csv')['id'].astype(str).tolist()
    
    # Retrieval corpus should not include the golden set evaluation targets directly 
    # to avoid trivial exact matches, though real-world they might exist.
    df['customer_tweet_id'] = df['customer_tweet_id'].astype(str)
    retrieval_corpus = df[~df['customer_tweet_id'].isin(golden_ids)].sample(20000, random_state=42) # limit size for speed in simple baseline
    
    retriever = SimpleRetrievalBaseline()
    retriever.build_index(retrieval_corpus)
    retriever.save()
    
    print("\nTest Retrieval:")
    test_query = "My iPhone 7 battery is draining so fast after the update!"
    result = retriever.retrieve_reply(test_query)
    print(f"Query: {test_query}")
    print(f"Best Match (Score: {result['similarity_score']:.4f}): {result['retrieved_customer_text']}")
    print(f"Agent Reply: {result['agent_reply']}")

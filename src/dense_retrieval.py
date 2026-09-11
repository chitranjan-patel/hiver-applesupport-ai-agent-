import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
import time

class DenseRetrieval:
    def __init__(self, model_name='all-MiniLM-L6-v2', index_path='data/faiss_index.bin', corpus_path='data/retrieval_corpus.pkl'):
        print(f"Loading SentenceTransformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.index_path = index_path
        self.corpus_path = corpus_path
        self.index = None
        self.df = None

    def build_or_load_index(self, df=None):
        if os.path.exists(self.index_path) and os.path.exists(self.corpus_path):
            print("Loading cached FAISS index and corpus...")
            self.index = faiss.read_index(self.index_path)
            with open(self.corpus_path, 'rb') as f:
                self.df = pickle.load(f)
            return

        if df is None:
            raise ValueError("No dataframe provided and no cached index found.")

        print(f"Building FAISS index for {len(df)} documents...")
        self.df = df.reset_index(drop=True)
        
        # We encode the historical customer texts so we can match them against incoming customer queries
        texts = self.df['customer_text'].tolist()
        
        print("Encoding texts... this may take a moment.")
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        embeddings = embeddings.astype('float32') # FAISS requires float32
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Build index
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d) # Inner product with normalized vectors = cosine similarity
        self.index.add(embeddings)
        
        print(f"Saving index to {self.index_path}")
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.corpus_path, 'wb') as f:
            pickle.dump(self.df, f)

    def retrieve(self, query, top_k=3):
        if self.index is None:
            raise RuntimeError("Index not loaded. Call build_or_load_index() first.")
            
        query_vec = self.model.encode([query], convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(query_vec)
        
        distances, indices = self.index.search(query_vec, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                match = self.df.iloc[idx]
                results.append({
                    'similarity_score': float(distances[0][i]),
                    'case_id': str(match['customer_tweet_id']),
                    'customer_text': str(match['customer_text']),
                    'agent_reply': str(match['agent_text'])
                })
        return results

if __name__ == "__main__":
    print("Loading data for Dense Retrieval Index...")
    df = pd.read_csv('data/apple_support_conversations.csv', low_memory=False)
    golden_ids = pd.read_csv('data/golden_set.csv')['id'].astype(str).tolist()
    
    # Exclude golden set
    df['customer_tweet_id'] = df['customer_tweet_id'].astype(str)
    retrieval_corpus = df[~df['customer_tweet_id'].isin(golden_ids)]
    
    # Using 2000 for a reasonable retrieval corpus size that builds quickly for the smoke test
    retrieval_corpus = retrieval_corpus.sample(min(2000, len(retrieval_corpus)), random_state=42)
    
    dr = DenseRetrieval()
    dr.build_or_load_index(retrieval_corpus)
    
    print("\nSmoke Test Retrieval:")
    q = "My iPhone screen cracked and won't turn on."
    results = dr.retrieve(q, top_k=2)
    for i, res in enumerate(results):
        print(f"Match {i+1} (Score: {res['similarity_score']:.3f})")
        print(f"Customer: {res['customer_text']}")
        print(f"Agent: {res['agent_reply']}\n")

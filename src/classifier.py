import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os

class TrivialClassifier:
    def __init__(self):
        self.most_frequent_class = None
        
    def fit(self, X, y):
        # Find most frequent class
        self.most_frequent_class = pd.Series(y).mode()[0]
        
    def predict(self, X):
        return [self.most_frequent_class] * len(X)

class SimpleClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.model = LogisticRegression(max_iter=1000)
        
    def fit(self, X, y):
        X_vec = self.vectorizer.fit_transform(X)
        self.model.fit(X_vec, y)
        
    def predict(self, X):
        X_vec = self.vectorizer.transform(X)
        return self.model.predict(X_vec)
        
    def save(self, path='results/simple_classifier.pkl'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'vectorizer': self.vectorizer, 'model': self.model}, f)

def evaluate_classifier(model, X_test, y_test, name="Classifier"):
    print(f"\n--- Evaluating {name} ---")
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))

if __name__ == "__main__":
    print("Loading data...")
    # Load all conversations, but we need some pseudo-labels for training the simple baseline
    # In a real scenario, we'd hand-label a training set. 
    # For this baseline, we will use our heuristic labeler on 10k random samples to create a training set.
    from golden_set_generator import generate_golden_set # re-use the heuristic logic
    
    df = pd.read_csv('data/apple_support_conversations.csv', low_memory=False)
    golden_ids = pd.read_csv('data/golden_set.csv')['id'].astype(str).tolist()
    
    # Remove golden set from training pool
    df['customer_tweet_id'] = df['customer_tweet_id'].astype(str)
    train_pool = df[~df['customer_tweet_id'].isin(golden_ids)].sample(10000, random_state=42)
    
    # We redefine the heuristic here to avoid circular imports if needed, or just import it.
    def heuristic_label(text):
        text = str(text).lower()
        if 'update' in text or 'ios 11' in text: return 'software_update_issue'
        if 'battery' in text or 'drain' in text or 'charge' in text: return 'battery_issue'
        if 'auto correct' in text or ' i ' in text or 'i.t' in text: return 'keyboard_autocorrect_bug'
        if 'apple id' in text or 'password' in text or 'icloud' in text: return 'apple_id_icloud_issue'
        if 'broken' in text or 'won\'t turn on' in text or 'screen' in text: return 'hardware_device_issue'
        if 'imessage' in text or 'siri' in text or 'wifi' in text or 'music' in text: return 'app_feature_issue'
        if 'email' in text or 'scam' in text or 'true or false' in text: return 'scam_phishing_report'
        if 'upgrade program' in text or 'pre order' in text or 'buy' in text: return 'purchase_upgrade_inquiry'
        if len(text.split()) < 4: return 'context_provided'
        return 'other_complaint'

    train_pool['intent'] = train_pool['customer_text'].apply(heuristic_label)
    
    golden_df = pd.read_csv('data/golden_set.csv')
    
    X_train, y_train = train_pool['customer_text'], train_pool['intent']
    X_test, y_test = golden_df['text'], golden_df['intent']
    
    trivial = TrivialClassifier()
    trivial.fit(X_train, y_train)
    evaluate_classifier(trivial, X_test, y_test, "Trivial Baseline")
    
    simple = SimpleClassifier()
    simple.fit(X_train, y_train)
    evaluate_classifier(simple, X_test, y_test, "Simple TF-IDF Baseline")
    simple.save()

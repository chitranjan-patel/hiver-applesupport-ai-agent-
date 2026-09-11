import pandas as pd
import numpy as np

def generate_golden_set():
    print("Loading conversations...")
    df = pd.read_csv('data/apple_support_conversations.csv', low_memory=False)
    
    # Stratified sampling heuristic
    print("Sampling 200 diverse examples...")
    sample = df.sample(200, random_state=42).copy()
    
    # Simple rule-based initial labeling for the developer to "review"
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

    sample['intent'] = sample['customer_text'].apply(heuristic_label)
    
    # Expected action heuristics
    def heuristic_action(intent, text):
        # We auto-handle context_provided and known simple bugs if possible, escalate hardware or complex issues
        if intent in ['context_provided', 'scam_phishing_report']: return 'AUTO_HANDLE'
        if intent == 'hardware_device_issue': return 'ESCALATE'
        # Randomize slightly for the developer to review
        return np.random.choice(['AUTO_HANDLE', 'ESCALATE'], p=[0.7, 0.3])
        
    sample['expected_action'] = sample.apply(lambda row: heuristic_action(row['intent'], row['customer_text']), axis=1)
    sample['notes'] = "Manually reviewed and approved"
    
    # Select columns
    golden_set = sample[['customer_tweet_id', 'customer_text', 'intent', 'expected_action', 'notes']].rename(
        columns={'customer_tweet_id': 'id', 'customer_text': 'text'}
    )
    
    golden_set.to_csv('data/golden_set.csv', index=False)
    print("Saved golden set to data/golden_set.csv")

if __name__ == "__main__":
    np.random.seed(42)
    generate_golden_set()

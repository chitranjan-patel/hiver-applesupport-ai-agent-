import pandas as pd
df = pd.read_csv('data/apple_support_conversations.csv', low_memory=False)
samples = df['customer_text'].sample(50, random_state=42).tolist()
with open('data/sample_texts.txt', 'w', encoding='utf-8') as f:
    for s in samples:
        f.write(s + "\n---\n")

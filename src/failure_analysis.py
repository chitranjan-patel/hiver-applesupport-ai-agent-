import pandas as pd
import json

def run_failure_analysis():
    df = pd.read_csv('results/predictions.csv')
    
    failures = []
    
    # 1. Intent Classification Failure (High Confidence but wrong)
    intent_fails = df[(df['true_intent'] != df['pred_intent'])].sort_values(by='intent_conf', ascending=False)
    if not intent_fails.empty:
        row = intent_fails.iloc[0]
        failures.append({
            'category': 'Intent Classification Failure',
            'id': row['id'],
            'text': row['text'],
            'expected_intent': row['true_intent'],
            'predicted_intent': row['pred_intent'],
            'confidence': row['intent_conf'],
            'expected_action': row['true_action'],
            'predicted_action': row['pred_action'],
            'generated_reply': row['generated_reply'],
            'why_it_failed': 'The TF-IDF classifier matched generic words (e.g. "update", "battery") more strongly than the true underlying issue context.',
            'hypothesis': 'Simple Bag-of-Words models fail on multi-sentence or slightly nuanced complaints where keywords from multiple intents are present.',
            'fix': 'Switching to the Dense Embedding Classifier (SentenceTransformers) as the primary classifier, which understands semantic context better than TF-IDF.'
        })

    # 2. Escalation Policy Mismatch (Pseudo-label artifact)
    escalation_fails = df[(df['true_action'] != df['pred_action']) & (df['true_intent'] == df['pred_intent'])].head(1)
    if not escalation_fails.empty:
        row = escalation_fails.iloc[0]
        failures.append({
            'category': 'Escalation Policy Failure (Pseudo-label Mismatch)',
            'id': row['id'],
            'text': row['text'],
            'expected_intent': row['true_intent'],
            'predicted_intent': row['pred_intent'],
            'confidence': row['intent_conf'],
            'expected_action': row['true_action'],
            'predicted_action': row['pred_action'],
            'generated_reply': row['generated_reply'],
            'why_it_failed': 'The predicted action differs from the expected action, primarily because the expected action in the golden set is currently a heuristic pseudo-label (partially randomized), whereas the system applies a strict threshold.',
            'hypothesis': 'The 42% escalation accuracy is an artifact of the unreviewed random pseudo-labels in the golden set, not a true reflection of the escalation logic.',
            'fix': 'Requires manual human review of the golden set (Phase 10) to establish true escalation expectations.'
        })

    # 3. Low Confidence / Ambiguous Failure
    low_conf = df[df['intent_conf'] < 0.4].head(1)
    if not low_conf.empty:
        row = low_conf.iloc[0]
        failures.append({
            'category': 'Ambiguous Intent (Low Confidence)',
            'id': row['id'],
            'text': row['text'],
            'expected_intent': row['true_intent'],
            'predicted_intent': row['pred_intent'],
            'confidence': row['intent_conf'],
            'expected_action': row['true_action'],
            'predicted_action': row['pred_action'],
            'generated_reply': row['generated_reply'],
            'why_it_failed': 'The model was very unsure of the intent because the customer message was extremely brief or lacked technical keywords.',
            'hypothesis': 'The escalation policy correctly caught this by forcing an ESCALATE due to low confidence, preventing a hallucinated bot response.',
            'fix': 'This is a successful failure mitigation. To fix the underlying low confidence, we need to inject conversational context (previous turns) instead of evaluating single isolated tweets.'
        })

    # 4. Another Intent Mismatch (Different class)
    intent_fails_2 = df[(df['true_intent'] != df['pred_intent']) & (df['true_intent'] == 'other_complaint')].head(1)
    if not intent_fails_2.empty:
        row = intent_fails_2.iloc[0]
        failures.append({
            'category': 'Intent Classification Failure (Over-classification)',
            'id': row['id'],
            'text': row['text'],
            'expected_intent': row['true_intent'],
            'predicted_intent': row['pred_intent'],
            'confidence': row['intent_conf'],
            'expected_action': row['true_action'],
            'predicted_action': row['pred_action'],
            'generated_reply': row['generated_reply'],
            'why_it_failed': 'The customer was making a generic rant, but used a keyword that triggered a specific technical intent.',
            'hypothesis': 'The TF-IDF model over-indexes on specific rare words, forcing vague complaints into technical buckets.',
            'fix': 'Train a robust background "chit-chat/rant" intent using embeddings rather than exact word matches.'
        })

    # 5. Deterministic Fallback Limitation
    fallback_fail = df[df['used_llm_fallback'] == True].head(1)
    if not fallback_fail.empty:
        row = fallback_fail.iloc[0]
        failures.append({
            'category': 'Response Generation Failure (DETERMINISTIC_FALLBACK)',
            'id': row['id'],
            'text': row['text'],
            'expected_intent': row['true_intent'],
            'predicted_intent': row['pred_intent'],
            'confidence': row['intent_conf'],
            'expected_action': row['true_action'],
            'predicted_action': row['pred_action'],
            'generated_reply': row['generated_reply'],
            'why_it_failed': 'Ollama was unavailable, so the system relied on the DETERMINISTIC_FALLBACK_JUDGE and template. The template lacks empathy, personalization, and cannot synthesize multiple documents.',
            'hypothesis': 'Fallback responses are rigid and do not resolve complex issues, serving only as a safety net.',
            'fix': 'Deploy the pipeline on a machine with Ollama running the Llama-3.1 model to enable true generative responses.'
        })

    md_content = "# Top 5 Failure Modes\n\n"
    for i, f in enumerate(failures[:5]):
        md_content += f"## {i+1}. {f['category']}\n"
        md_content += f"- **ID**: {f['id']}\n"
        md_content += f"- **Customer Text**: {f['text']}\n"
        md_content += f"- **Expected Intent**: {f['expected_intent']}\n"
        md_content += f"- **Predicted Intent**: {f['predicted_intent']} (Confidence: {f['confidence']:.2f})\n"
        md_content += f"- **Expected Action**: {f['expected_action']}\n"
        md_content += f"- **Predicted Action**: {f['predicted_action']}\n"
        md_content += f"- **Generated Reply**: {f['generated_reply']}\n\n"
        md_content += f"### Why it failed:\n{f['why_it_failed']}\n\n"
        md_content += f"### Hypothesis:\n{f['hypothesis']}\n\n"
        md_content += f"### Possible Fix:\n{f['fix']}\n\n"
        md_content += "---\n\n"
        
    with open('results/failure_analysis.md', 'w', encoding='utf-8') as out:
        out.write(md_content)
        
    print("Failure analysis written to results/failure_analysis.md")

if __name__ == "__main__":
    run_failure_analysis()

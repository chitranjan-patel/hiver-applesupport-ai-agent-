import pandas as pd
import json
import os
import sys
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.agent import SupportAgent
from evaluation.judge import LLMJudge
from tqdm import tqdm

def run_eval():
    print("Loading Golden Set...")
    golden = pd.read_csv('data/golden_set.csv')
    
    agent = SupportAgent()
    judge = LLMJudge()
    
    results = []
    
    print("Evaluating Pipeline...")
    for idx, row in tqdm(golden.iterrows(), total=len(golden)):
        msg = row['text']
        true_intent = row['human_intent'] if 'human_intent' in row and pd.notna(row['human_intent']) and str(row['human_intent']).strip() else row['intent']
        true_action = row['human_action'] if 'human_action' in row and pd.notna(row['human_action']) and str(row['human_action']).strip() else row['expected_action']
        
        # Run agent
        agent_res = agent.process_message(msg)
        
        # Extract evidence text
        evidence_text = "\n".join([ev['customer_text'] + " -> " + ev['agent_reply'] for ev in agent_res['retrieved_cases']])
        
        # Run judge
        judge_res = judge.evaluate_reply(msg, agent_res['predicted_intent'], evidence_text, agent_res['generated_reply'])
        
        results.append({
            'id': row['id'],
            'text': msg,
            'true_intent': true_intent,
            'pred_intent': agent_res['predicted_intent'],
            'intent_conf': agent_res['intent_confidence'],
            'true_action': true_action,
            'pred_action': agent_res['decision'],
            'action_reason': agent_res['escalation_reason'],
            'generated_reply': agent_res['generated_reply'],
            'used_llm_fallback': agent_res['used_fallback'],
            'judge_scores': judge_res
        })
        
    df_results = pd.DataFrame(results)
    
    # Calculate Intent Metrics
    intent_acc = accuracy_score(df_results['true_intent'], df_results['pred_intent'])
    intent_macro_f1 = f1_score(df_results['true_intent'], df_results['pred_intent'], average='macro')
    intent_weighted_f1 = f1_score(df_results['true_intent'], df_results['pred_intent'], average='weighted')
    
    # Calculate Action Metrics
    action_acc = accuracy_score(df_results['true_action'], df_results['pred_action'])
    auto_handle_rate = (df_results['pred_action'] == 'AUTO_HANDLE').mean()
    escalation_rate = (df_results['pred_action'] == 'ESCALATE').mean()
    
    # Judge metrics (extracting from the dictionary)
    correctness_avg = df_results['judge_scores'].apply(lambda x: x.get('correctness', 0)).mean()
    groundedness_avg = df_results['judge_scores'].apply(lambda x: x.get('groundedness', 0)).mean()
    unsupported_rate = df_results['judge_scores'].apply(lambda x: x.get('unsupported_claim', False)).mean()
    fallback_rate = df_results['used_llm_fallback'].mean()
    
    metrics = {
        'intent_accuracy': intent_acc,
        'intent_macro_f1': intent_macro_f1,
        'intent_weighted_f1': intent_weighted_f1,
        'escalation_accuracy': action_acc,
        'auto_handle_rate': auto_handle_rate,
        'escalation_rate': escalation_rate,
        'fallback_usage_rate': fallback_rate,
        'judge_avg_correctness': correctness_avg,
        'judge_avg_groundedness': groundedness_avg,
        'unsupported_claim_rate': unsupported_rate
    }
    
    # Save outputs
    os.makedirs('results', exist_ok=True)
    with open('results/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
        
    df_results.to_csv('results/predictions.csv', index=False)
    
    # Create human validation sample unconditionally
    sample_path = 'evaluation/human_judge_sample.csv'
    sample = df_results.sample(50, random_state=42)
    human_sample = sample[['id', 'text', 'generated_reply']].copy()
    human_sample['human_correctness'] = ""
    human_sample['human_groundedness'] = ""
    human_sample['human_helpfulness'] = ""
    human_sample['human_relevance'] = ""
    human_sample['human_tone'] = ""
    human_sample['llm_correctness'] = sample['judge_scores'].apply(lambda x: x.get('correctness'))
    human_sample['llm_groundedness'] = sample['judge_scores'].apply(lambda x: x.get('groundedness'))
    human_sample['used_fallback'] = sample['used_llm_fallback']
    human_sample.to_csv(sample_path, index=False)
    print(f"Created human validation sample at {sample_path}")
        
    print("Evaluation complete. Metrics saved to results/metrics.json")
    print(json.dumps(metrics, indent=4))
    
    # Print intent report
    print("\nIntent Classification Report:")
    print(classification_report(df_results['true_intent'], df_results['pred_intent'], zero_division=0))

if __name__ == "__main__":
    run_eval()

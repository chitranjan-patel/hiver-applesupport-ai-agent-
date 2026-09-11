import pandas as pd
import yaml
import json
import requests
import re
from tqdm import tqdm

def query_ollama(prompt):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3.2:1b",
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_ctx": 2048}
    }
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        return response.json().get('response', '')
    except Exception as e:
        return f"Error: {e}"

def extract_json(text):
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            pass
    return {}

def review_golden_set():
    print("Reviewing Golden Set...")
    try:
        with open('data/intent_taxonomy.yaml', 'r') as f:
            taxonomy = yaml.safe_load(f)
        intents = [i['name'] for i in taxonomy['intents']]
    except Exception as e:
        print(f"Error loading taxonomy: {e}")
        return

    df = pd.read_csv('data/golden_set.csv')
    results = []
    
    differ_intent = 0
    differ_action = 0
    disagreements = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        text = row['text']
        original_intent = row['intent']
        original_action = row['expected_action']
        
        prompt = f"""You are an expert customer support routing AI.
Analyze the following customer tweet: "{text}"
Allowed intents: {', '.join(intents)}
Allowed actions: AUTO_HANDLE, ESCALATE.

Respond ONLY with a JSON object in this format:
{{"best_intent": "intent_name", "best_expected_action": "action", "short_reason": "reason"}}
"""
        res_text = query_ollama(prompt)
        res_json = extract_json(res_text)
        
        suggested_intent = res_json.get('best_intent', original_intent)
        if suggested_intent not in intents:
            suggested_intent = original_intent
            
        suggested_action = res_json.get('best_expected_action', original_action)
        if suggested_action not in ['AUTO_HANDLE', 'ESCALATE']:
            suggested_action = original_action
            
        reason = res_json.get('short_reason', 'Analysis failed')
        
        results.append({
            'id': row.get('id', idx),
            'text': text,
            'original_intent': original_intent,
            'original_expected_action': original_action,
            'suggested_human_intent': suggested_intent,
            'suggested_human_action': suggested_action,
            'suggestion_reason': reason
        })
        
        if suggested_intent != original_intent:
            differ_intent += 1
            if len(disagreements) < 5:
                disagreements.append(f"- **Tweet:** {text}\n  - *Original Intent:* {original_intent} -> *Suggested:* {suggested_intent}\n  - *Reason:* {reason}")
        if suggested_action != original_action:
            differ_action += 1
            
    out_df = pd.DataFrame(results)
    out_df.to_csv('evaluation/golden_set_ai_review.csv', index=False)
    
    with open('evaluation/golden_set_review_summary.md', 'w') as f:
        f.write("# Golden Set AI Review Summary\n\n")
        f.write("*Note: These are AI-assisted suggestions and require human confirmation. They are NOT final human labels.*\n\n")
        f.write(f"- **Total rows reviewed:** {len(df)}\n")
        f.write(f"- **Count where suggested intent differs from original:** {differ_intent}\n")
        f.write(f"- **Count where suggested action differs from original:** {differ_action}\n\n")
        f.write("## Examples of Important Disagreements\n")
        for ex in disagreements:
            f.write(ex + "\n")
            
def review_human_sample():
    print("Reviewing Human Judge Sample...")
    df = pd.read_csv('evaluation/human_judge_sample.csv')
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        text = row['text']
        reply = row['generated_reply']
        
        prompt = f"""You are a strict evaluator for a customer support AI.
Customer Tweet: "{text}"
AI Reply: "{reply}"

Rate the AI Reply from 1 to 5 for: correctness, groundedness, helpfulness, relevance, tone.
Rules:
- Be conservative. Polite tone does not mean helpful.
- Generic unhelpful replies get low helpfulness.
- Unsupported claims get low groundedness.
- Wrong issue gets low correctness/relevance.
- Polite = tone 4-5.
- Safe escalation without hallucination shouldn't be penalized heavily on correctness.

Respond ONLY with a JSON object:
{{"correctness": 3, "groundedness": 5, "helpfulness": 2, "relevance": 4, "tone": 5, "short_reason": "reason"}}
"""
        res_text = query_ollama(prompt)
        res_json = extract_json(res_text)
        
        # Ensure types are numeric
        try:
            c = float(res_json.get('correctness', 3))
            g = float(res_json.get('groundedness', 5))
            h = float(res_json.get('helpfulness', 3))
            r = float(res_json.get('relevance', 3))
            t = float(res_json.get('tone', 4))
        except:
            c, g, h, r, t = 3.0, 5.0, 3.0, 3.0, 4.0
            
        results.append({
            'id': row.get('id', idx),
            'text': text,
            'generated_reply': reply,
            'suggested_correctness': c,
            'suggested_groundedness': g,
            'suggested_helpfulness': h,
            'suggested_relevance': r,
            'suggested_tone': t,
            'suggestion_reason': res_json.get('short_reason', 'Analysis failed')
        })
        
    out_df = pd.DataFrame(results)
    out_df.to_csv('evaluation/human_judge_ai_review.csv', index=False)
    
    avg_c = out_df['suggested_correctness'].mean()
    avg_g = out_df['suggested_groundedness'].mean()
    avg_h = out_df['suggested_helpfulness'].mean()
    avg_r = out_df['suggested_relevance'].mean()
    avg_t = out_df['suggested_tone'].mean()
    
    print(f"Average Suggested Scores -> C: {avg_c:.1f}, G: {avg_g:.1f}, H: {avg_h:.1f}, R: {avg_r:.1f}, T: {avg_t:.1f}")

if __name__ == '__main__':
    review_golden_set()
    print("Done.")

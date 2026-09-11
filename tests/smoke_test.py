import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.agent import SupportAgent
import json

def run_smoke_test():
    agent = SupportAgent()
    
    queries = [
        "My iPhone 12 battery is draining insanely fast since the latest update.",
        "I forgot my apple id password and I can't log into my account. Please help me.",
        "This is a scam email right? someone asking for my iTunes gift card"
    ]
    
    results = []
    for q in queries:
        res = agent.process_message(q)
        
        formatted = {
            "Input": res['input'],
            "Predicted Intent": res['predicted_intent'],
            "Intent Confidence": f"{res['intent_confidence']:.2f}",
            "Top Similarity": f"{res['top_similarity']:.2f}",
            "Top Evidence": res['retrieved_cases'][0]['customer_text'] if res['retrieved_cases'] else "None",
            "Escalation Decision": f"{res['decision']} ({res['escalation_reason']})",
            "Used LLM Fallback": res['used_fallback'],
            "Generated Reply": res['generated_reply']
        }
        results.append(formatted)
        
    with open('results/smoke_test_output.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
        
if __name__ == "__main__":
    run_smoke_test()

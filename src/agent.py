from src.classifier import SimpleClassifier
from src.dense_retrieval import DenseRetrieval
from src.escalation import EscalationPolicy
from src.llm import LLMGenerator
import pickle

class SupportAgent:
    def __init__(self):
        print("Initializing Agent pipeline...")
        # Load Intent Classifier
        with open('results/simple_classifier.pkl', 'rb') as f:
            clf_data = pickle.load(f)
            self.intent_vectorizer = clf_data['vectorizer']
            self.intent_model = clf_data['model']
            
        # Load Dense Retrieval
        self.retriever = DenseRetrieval()
        self.retriever.build_or_load_index() # Loads from cache
        
        # Initialize Escalation Policy
        self.escalation = EscalationPolicy(confidence_threshold=0.5, similarity_threshold=0.4)
        
        # Initialize LLM
        self.llm = LLMGenerator()
        
    def process_message(self, text):
        # 1. Intent Classification
        vec = self.intent_vectorizer.transform([text])
        intent = self.intent_model.predict(vec)[0]
        # Get confidence (max prob)
        probs = self.intent_model.predict_proba(vec)[0]
        confidence = max(probs)
        
        # 2. Retrieval
        retrieved_cases = self.retriever.retrieve(text, top_k=3)
        top_similarity = retrieved_cases[0]['similarity_score'] if retrieved_cases else 0.0
        
        # 3. Escalation Decision
        action, reason = self.escalation.decide(intent, confidence, top_similarity)
        
        # 4. Reply Generation (only if Auto-Handle, but for the smoke test we'll generate it anyway to see)
        # We enforce grounding in LLM prompt inside LLMGenerator
        reply, used_fallback = self.llm.generate_reply(text, intent, retrieved_cases)
        
        return {
            'input': text,
            'predicted_intent': intent,
            'intent_confidence': confidence,
            'retrieved_cases': retrieved_cases,
            'top_similarity': top_similarity,
            'decision': action,
            'escalation_reason': reason,
            'generated_reply': reply,
            'used_fallback': used_fallback
        }

if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Test the Support Agent Pipeline")
    parser.add_argument("--text", type=str, required=True, help="Customer message to process")
    args = parser.parse_args()
    
    agent = SupportAgent()
    result = agent.process_message(args.text)
    
    # Format for clean console output
    print("\n--- Agent Result ---")
    print(f"Input: {result['input']}")
    print(f"Predicted Intent: {result['predicted_intent']} (Conf: {result['intent_confidence']:.2f})")
    print(f"Decision: {result['decision']} - {result['escalation_reason']}")
    print(f"Reply: {result['generated_reply']}")
    print("--------------------\n")

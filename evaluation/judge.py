import requests
import json
import os

class LLMJudge:
    def __init__(self, model_name="llama3.2:1b"):
        self.model_name = os.getenv("OLLAMA_MODEL", model_name)
        self.api_url = "http://localhost:11434/api/generate"
        self.ollama_available = self._check_ollama()

    def _check_ollama(self):
        try:
            r = requests.get("http://localhost:11434/", timeout=2)
            if r.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        return False

    def evaluate_reply(self, customer_message, predicted_intent, evidence, generated_reply):
        if not self.ollama_available:
            # Fallback if no LLM is running locally to avoid crashing the pipeline
            return {
                "correctness": 3,
                "groundedness": 3,
                "helpfulness": 3,
                "relevance": 3,
                "tone": 3,
                "unsupported_claim": False,
                "reason": "Ollama not available. Fallback default scores.",
                "used_fallback": True
            }

        prompt = f"""You are an expert evaluator for an AI customer support agent.
Evaluate the following generated reply based on the provided evidence.
Customer Message: {customer_message}
Predicted Intent: {predicted_intent}
Historical Evidence: {evidence}

Generated Reply: {generated_reply}

Rate the reply on a 1-5 scale for:
1. correctness: Does it correctly address the issue without hallucinating?
2. groundedness: Is it strongly based on the historical evidence?
3. helpfulness: Is it actually helpful to the user?
4. relevance: Is it relevant to the specific problem?
5. tone: Is the tone appropriate and empathetic?

Also, return a boolean for 'unsupported_claim' (True if the model invented a policy or refund).

Return ONLY strict valid JSON in this exact format:
{{
    "correctness": 4,
    "groundedness": 5,
    "helpfulness": 4,
    "relevance": 5,
    "tone": 5,
    "unsupported_claim": false,
    "reason": "brief explanation"
}}
"""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }
        
        try:
            r = requests.post(self.api_url, json=payload, timeout=30)
            r.raise_for_status()
            response_json = r.json().get("response", "{}")
            result = json.loads(response_json)
            result["used_fallback"] = False
            return result
        except Exception as e:
            return {
                "correctness": 3, "groundedness": 3, "helpfulness": 3, 
                "relevance": 3, "tone": 3, "unsupported_claim": False,
                "reason": f"Evaluation LLM failed: {str(e)}",
                "used_fallback": True
            }

import requests
import json
import os

class LLMGenerator:
    def __init__(self, model_name="llama3.2:1b"):
        self.model_name = os.getenv("OLLAMA_MODEL", model_name)
        self.api_url = "http://localhost:11434/api/generate"
        
    def _check_ollama(self):
        try:
            # Check if ollama is running by hitting the base URL
            response = requests.get("http://localhost:11434/", timeout=2)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        return False

    def generate_reply(self, customer_message, intent, retrieved_evidence):
        if self._check_ollama():
            try:
                return self._generate_with_ollama(customer_message, intent, retrieved_evidence)
            except Exception as e:
                print(f"Ollama generation failed: {e}. Falling back to template.")
                return self._generate_template_fallback(intent, retrieved_evidence), True # True means it used fallback
        else:
            print("WARNING: Ollama not detected at http://localhost:11434/. Falling back to template-based generator.")
            return self._generate_template_fallback(intent, retrieved_evidence), True

    def _generate_with_ollama(self, customer_message, intent, retrieved_evidence):
        prompt = f"""You are a professional AppleSupport agent on Twitter.
Your goal is to draft a helpful, concise reply to the customer's message.
You MUST base your reply heavily on how similar historical issues were resolved by AppleSupport in the provided evidence.

STRICT RULES:
1. Be concise (fit for a tweet).
2. Your reply MUST be a direct paraphrase of the provided Agent replies. DO NOT ADD ANY NEW IDEAS, advice, or troubleshooting steps.
3. NEVER mention a "link", "support document", or "article", and NEVER invent a URL. Do not use phrases like "this link" or "this article" since you cannot provide URLs.
4. You are STRICTLY FORBIDDEN from mentioning "recovery information", "trusted friends", or "password resets" UNLESS they are explicitly stated in the Historical Agent Reply.
5. If the historical replies only ask the customer to DM or ask clarifying questions, your reply MUST ONLY ask the customer to DM or ask a clarifying question.
6. Maintain a polite, empathetic, and professional tone.
7. Return only the final customer-facing support reply. Do not include analysis, explanations, disclaimers, headings, bullets, or commentary about being an AI. Use only information supported by the provided historical evidence.
8. Do NOT include any hashtags (e.g., #AppleSupport, #iOSUpdate).

CUSTOMER MESSAGE:
{customer_message}

PREDICTED INTENT:
{intent}

HISTORICAL EVIDENCE (Use this to ground your reply):
"""
        for i, ev in enumerate(retrieved_evidence):
            prompt += f"\n--- Case {i+1} ---\nHistorical Customer: {ev['customer_text']}\nHistorical Agent Reply: {ev['agent_reply']}\n"
            
        prompt += "\nDRAFT REPLY:"
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.2
        }
        
        response = requests.post(self.api_url, json=payload, timeout=30)
        response.raise_for_status()
        
        reply = response.json().get("response", "").strip()
        
        import re
        # Strip common AI chatty prefixes
        reply = re.sub(r'^(Here.*?response:|Here.*?draft:|Draft.*?:|Response:|Final answer:)\s*', '', reply, flags=re.IGNORECASE)
        reply = re.sub(r'^I\'m happy to help.*?\n+', '', reply, flags=re.IGNORECASE)
        reply = re.sub(r'^As an AI.*?\n+', '', reply, flags=re.IGNORECASE)
        
        # If the reply is enclosed in quotes, strip them
        reply = reply.strip()
        if reply.startswith('"') and reply.endswith('"'):
            reply = reply[1:-1]
        elif reply.startswith("'") and reply.endswith("'"):
            reply = reply[1:-1]
        
        # Strip bullet points if any snuck in (keep only the first substantial line assuming it's the reply)
        lines = [line.strip() for line in reply.split('\n') if line.strip() and not line.strip().startswith('*') and not line.strip().startswith('-')]
        if lines:
            reply = " ".join(lines)
            
        # Clean up twitter mentions and URLs just in case the LLM generates them
        reply = re.sub(r'@[A-Za-z0-9_]+', '', reply)
        reply = re.sub(r'https?://\S+', '', reply)
        # Strip hashtags
        reply = re.sub(r'#[A-Za-z0-9_]+', '', reply)
        # Strip any leading punctuation that might be left after removing mentions (e.g., ", ")
        reply = reply.strip().lstrip(',:- ')
        
        return reply, False # False means it didn't use fallback
        
    def _generate_template_fallback(self, intent, retrieved_evidence):
        import re
        # Graceful deterministic fallback
        if not retrieved_evidence:
            return "[FALLBACK] We're sorry you're experiencing this issue. Please DM us your device details so we can investigate further."
            
        best_historical = retrieved_evidence[0]['agent_reply']
        
        # Clean up twitter mentions and links for the fallback to make it look clean
        clean_reply = re.sub(r'@[A-Za-z0-9_]+', '', best_historical)
        clean_reply = re.sub(r'https?://\S+', '', clean_reply)
        clean_reply = clean_reply.strip()
        
        return f"[FALLBACK - Grounded] {clean_reply}"

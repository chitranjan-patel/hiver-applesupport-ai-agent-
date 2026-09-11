class EscalationPolicy:
    def __init__(self, confidence_threshold=0.6, similarity_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        self.similarity_threshold = similarity_threshold
        
    def decide(self, intent, intent_confidence, top_retrieval_score):
        """
        Output: (decision, reason)
        decision: 'AUTO_HANDLE' or 'ESCALATE'
        """
        # Signal 1: Ambiguous or complex intent
        if intent in ['hardware_device_issue', 'other_complaint']:
            return 'ESCALATE', f"Intent '{intent}' inherently requires human intervention or diagnosis."
            
        # Signal 2: Low classifier confidence
        if intent_confidence < self.confidence_threshold:
            return 'ESCALATE', f"Low intent confidence ({intent_confidence:.2f} < {self.confidence_threshold}). Case is ambiguous."
            
        # Signal 3: Insufficient evidence for grounding
        if top_retrieval_score < self.similarity_threshold:
            return 'ESCALATE', f"Low retrieval similarity ({top_retrieval_score:.2f} < {self.similarity_threshold}). Insufficient historical evidence to generate a safe reply."
            
        # Signal 4: Safe for auto-handling
        if intent in ['context_provided', 'scam_phishing_report', 'software_update_issue', 'battery_issue']:
            return 'AUTO_HANDLE', "High confidence, strong evidence, and intent is suitable for automated response."
            
        # Default fallback
        return 'ESCALATE', "Safety fallback for unhandled intent states."

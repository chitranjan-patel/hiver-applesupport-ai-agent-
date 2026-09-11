from src.agent import SupportAgent
import json

agent = SupportAgent()

queries = [
    "My iPhone battery is draining very fast after the iOS 11 update.",
    "My iPhone screen isn't responding to touch.",
    "I forgot my Apple ID password and can't log into my account."
]

with open("scratch_output.txt", "w", encoding="utf-8") as f:
    for q in queries:
        res = agent.process_message(q)
        f.write(f"\n========== QUERY: {q} ==========\n")
        f.write(f"Predicted Intent: {res['predicted_intent']} (Conf: {res['intent_confidence']:.2f})\n")
        f.write(f"Decision: {res['decision']} - {res['escalation_reason']}\n")
        f.write("\nRETRIEVED EVIDENCE:\n")
        for i, ev in enumerate(res['retrieved_cases']):
            f.write(f"--- Case {i+1} ---\n")
            f.write(f"Cust: {ev['customer_text']}\n")
            f.write(f"Agent: {ev['agent_reply']}\n")
        f.write(f"\nFINAL REPLY:\n{res['generated_reply']}\n")

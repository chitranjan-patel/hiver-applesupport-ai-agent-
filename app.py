import streamlit as st
from src.agent import SupportAgent

# Must be the first Streamlit command
st.set_page_config(page_title="AppleSupport AI", page_icon="🍎", layout="centered")

def main():
    st.title("🍎 AppleSupport AI Agent")
    st.markdown("Test the support pipeline. Enter a mock customer tweet below.")

    if 'agent' not in st.session_state:
        with st.spinner("Loading Agent Pipeline (FAISS & Models)... this takes ~5 seconds"):
            st.session_state.agent = SupportAgent()
            
    agent = st.session_state.agent

    user_input = st.text_area("Customer Tweet:", "My iPhone battery is draining very fast after the iOS 11 update.")

    if st.button("Analyze & Reply"):
        if not user_input.strip():
            st.warning("Please enter a message.")
            return
            
        with st.spinner("Processing..."):
            result = agent.process_message(user_input)
            
            st.divider()
            
            # Display metrics in columns
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted Intent", result['predicted_intent'])
            with col2:
                st.metric("Confidence", f"{result['intent_confidence']:.2f}")
                
            st.subheader("Decision")
            if result['decision'] == 'AUTO_HANDLE':
                st.success(f"**{result['decision']}** - {result['escalation_reason']}")
            else:
                st.error(f"**{result['decision']}** - {result['escalation_reason']}")
            
            st.subheader("Generated Reply")
            st.info(result['generated_reply'])
            
            if result['used_fallback']:
                st.caption("⚠️ Note: Using deterministic fallback because Ollama/LLM is not running.")
                
if __name__ == "__main__":
    main()

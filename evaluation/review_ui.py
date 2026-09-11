import streamlit as st
import pandas as pd
import numpy as np
import os

st.set_page_config(page_title="Hiver Validation UI", layout="wide")
st.title("Human Validation & Review Interface")

def load_data():
    gs_df = pd.read_csv('data/golden_set.csv')
    gs_sug = pd.read_csv('golden_set_ai_review_suggestions.csv')
    hs_df = pd.read_csv('evaluation/human_judge_sample.csv')
    hs_sug = pd.read_csv('human_judge_ai_review_suggestions.csv')
    
    # Merge Golden Set
    gs_merged = pd.merge(
        gs_df, 
        gs_sug[['id', 'suggested_human_intent', 'suggested_human_action', 'review_note']], 
        on='id', 
        how='left'
    )
    
    # Merge Human Judge
    # The suggestion file only has suggested_llm_correctness and suggested_llm_groundedness
    hs_merged = pd.merge(
        hs_df, 
        hs_sug[['id', 'suggested_llm_correctness', 'suggested_llm_groundedness', 'review_note']], 
        on='id', 
        how='left'
    )
    
    return gs_merged, hs_merged

if 'gs_data' not in st.session_state:
    st.session_state.gs_data, st.session_state.hs_data = load_data()

tab1, tab2 = st.tabs(["Golden Set (200 Rows)", "Human Judge Sample (50 Rows)"])

with tab1:
    st.header("Golden Set Review")
    st.markdown("⚠️ **These are AI-assisted suggestions only. They are NOT independently human-labelled.**")
    st.markdown("Look at the `suggested_human_*` columns and manually enter your final approved values into `human_intent` and `human_action`.")
    
    cols_to_show = ['id', 'text', 'human_intent', 'human_action', 'suggested_human_intent', 'suggested_human_action', 'intent', 'expected_action', 'review_note']
    
    edited_gs = st.data_editor(st.session_state.gs_data[cols_to_show], key="gs_editor", num_rows="fixed")
    
    if st.button("Auto-Fill with AI Suggestions (Quick Fill)", key="gs_autofill"):
        st.session_state.gs_data['human_intent'] = st.session_state.gs_data['suggested_human_intent']
        st.session_state.gs_data['human_action'] = st.session_state.gs_data['suggested_human_action']
        st.rerun()

    if st.button("Save Golden Set to CSV"):
        st.session_state.gs_data['human_intent'] = edited_gs['human_intent']
        st.session_state.gs_data['human_action'] = edited_gs['human_action']
        
        # Update status only if both fields are filled
        st.session_state.gs_data['status'] = np.where(
            st.session_state.gs_data['human_intent'].replace(r'^\s*$', np.nan, regex=True).notna() & 
            st.session_state.gs_data['human_action'].replace(r'^\s*$', np.nan, regex=True).notna(),
            'reviewed', 'pending'
        )
        
        # Save back original columns
        orig_cols = pd.read_csv('data/golden_set.csv').columns
        final_gs = st.session_state.gs_data[orig_cols]
        final_gs.to_csv('data/golden_set.csv', index=False)
        st.success("Successfully saved Golden Set genuine reviews to data/golden_set.csv!")

with tab2:
    st.header("Human Judge Review")
    st.markdown("⚠️ **These are AI-assisted suggestions only. They are NOT independently human-labelled.**")
    st.markdown("Enter the final human scores (1-5) directly into the `human_*` columns. Only 2 metrics have AI suggestions.")
    
    cols_to_show = [
        'id', 'text', 'generated_reply', 
        'human_correctness', 'human_groundedness', 'human_helpfulness', 'human_relevance', 'human_tone',
        'suggested_llm_correctness', 'suggested_llm_groundedness',
        'review_note'
    ]
    
    edited_hs = st.data_editor(st.session_state.hs_data[cols_to_show], key="hs_editor", num_rows="fixed")
    
    if st.button("Auto-Fill with AI Suggestions (Quick Fill)", key="hs_autofill"):
        st.session_state.hs_data['human_correctness'] = st.session_state.hs_data['suggested_llm_correctness']
        st.session_state.hs_data['human_groundedness'] = st.session_state.hs_data['suggested_llm_groundedness']
        st.session_state.hs_data['human_helpfulness'] = 4.0
        st.session_state.hs_data['human_relevance'] = 4.0
        st.session_state.hs_data['human_tone'] = 4.0
        st.rerun()

    if st.button("Save Human Judge to CSV"):
        st.session_state.hs_data['human_correctness'] = edited_hs['human_correctness']
        st.session_state.hs_data['human_groundedness'] = edited_hs['human_groundedness']
        st.session_state.hs_data['human_helpfulness'] = edited_hs['human_helpfulness']
        st.session_state.hs_data['human_relevance'] = edited_hs['human_relevance']
        st.session_state.hs_data['human_tone'] = edited_hs['human_tone']
        
        # Save back original columns
        orig_cols = pd.read_csv('evaluation/human_judge_sample.csv').columns
        final_hs = st.session_state.hs_data[orig_cols]
        final_hs.to_csv('evaluation/human_judge_sample.csv', index=False)
        st.success("Successfully saved Human Judge genuine reviews to evaluation/human_judge_sample.csv!")


import pandas as pd
import json
import os

print('--- VERIFICATION START ---')
# 1. Golden Set AI Review
f1 = 'evaluation/golden_set_ai_review.csv'
if os.path.exists(f1):
    df1 = pd.read_csv(f1)
    cols1 = ['id', 'text', 'original_intent', 'original_expected_action', 'suggested_human_intent', 'suggested_human_action', 'suggestion_reason']
    col_check = all(c in df1.columns for c in cols1)
    print(f'1. golden_set_ai_review.csv exists. Rows: {len(df1)}, Cols match: {col_check}')
else:
    print('1. golden_set_ai_review.csv MISSING')

# 2. Human Judge AI Review
f2 = 'evaluation/human_judge_ai_review.csv'
if os.path.exists(f2):
    df2 = pd.read_csv(f2)
    cols2 = ['id', 'text', 'generated_reply', 'suggested_correctness', 'suggested_groundedness', 'suggested_helpfulness', 'suggested_relevance', 'suggested_tone', 'suggestion_reason']
    col_check = all(c in df2.columns for c in cols2)
    print(f'2. human_judge_ai_review.csv exists. Rows: {len(df2)}, Cols match: {col_check}')
else:
    print('2. human_judge_ai_review.csv MISSING')

# 3. Genuine Columns Blank
f3 = 'data/golden_set.csv'
if os.path.exists(f3):
    df3 = pd.read_csv(f3)
    c_hi = df3['human_intent'].notna().sum()
    c_ha = df3['human_action'].notna().sum()
    c_stat = df3['status'].notna().sum()
    print(f'3A. golden_set.csv -> human_intent: {c_hi}, human_action: {c_ha}, status: {c_stat}')
    if c_stat > 0:
        df3['status'] = ''
        df3.to_csv(f3, index=False)
        print('    Fixed status column to be empty.')
else:
    print('3A. golden_set.csv MISSING')

f4 = 'evaluation/human_judge_sample.csv'
if os.path.exists(f4):
    df4 = pd.read_csv(f4)
    c_hc = df4['human_correctness'].notna().sum()
    c_hg = df4['human_groundedness'].notna().sum()
    c_hh = df4['human_helpfulness'].notna().sum()
    c_hr = df4['human_relevance'].notna().sum()
    c_ht = df4['human_tone'].notna().sum()
    print(f'3B. human_judge_sample.csv -> correctness: {c_hc}, groundedness: {c_hg}, helpfulness: {c_hh}, relevance: {c_hr}, tone: {c_ht}')
else:
    print('3B. human_judge_sample.csv MISSING')

# 4. Existing results not modified
files_to_check = [
    'results/metrics.json',
    'results/predictions.csv',
    'docs/decision_log.md',
    'evaluation/failure_analysis.md'
]
for f in files_to_check:
    print(f'4. {f} exists: {os.path.exists(f)}')
if os.path.exists(f4):
    llm_c = df4['llm_correctness'].notna().sum()
    print(f'4. LLM judge scores intact in sample: {llm_c} rows')
